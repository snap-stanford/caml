import sys
import os
sys.path.append(os.getcwd())

import gc
import argparse
import wandb
from inspect import signature
import copy

# TODO: clean up these imports from notebook (many not necessary)
import os
import gzip
import json
import scipy
import random
import math
import numpy as np
from string import ascii_letters
import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict, KFold, train_test_split
from scipy.sparse import hstack
import pickle
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from IPython import embed
from pytorch_lightning.loggers import WandbLogger
import sys
from typing import Union
from scipy.sparse import csr_matrix, coo_matrix
from laboratory.building_blocks.utils import get_optimizer_scheduler

# Custom modules:
from causal_singletask.consts_and_utils import *
import src.models
from src import models
from causal_singletask.torch_utils import BestValueLogger, EarlyStopping
from metalearn.eval import treatmentRecall, treatmentPrecision, addLogDict, treatmentRATE, treatmentChoose2, countPositiveLabels
from metalearn.parse import PatientDataLoader, XLearnerM0, QueryPatientsByDrug
from metalearn.caml import camlAdapter
from metalearn.task_utils import (TaskEmbeddingMapper, ClaimsTaskDataset,
    task_embedding_paths, sparse_batch_collate, sparse_batch_collate2,sparse_batch_collate3, filter_for_used_drugs, sparse_csr_to_tensor, SINTaskDataset)
from pdb import set_trace as bp
from causal_singletask.models.xlearner import (XTrtLearner, TLearner,XLearner, RLearner)

import catenets.logger as log
import signal
import time
import catenets.models.constants


import importlib


def handler(signum, frame):
    # This function will be called when the timer expires
    raise SystemExit("Script ran for more than 60 minutes, exiting.")


def main(args):
    """ Training script for deep models-based for CATE predition
        All arguments to the main() function are hyperparameters that can be set in the wandb sweep
    """

    test_str =""
    if args.TEST_MODE:
        test_str = "-test"
        performance_logs = {}

    run = wandb.init(config=args,
        project="dl-zero-shot-cate" + test_str,
        entity='dsp'
    )

    logWANDBRun(run)
    log.add(sys.stdout,level=0)

    if args.x_trt_learner or args.t_learner or args.x_learner or args.r_learner or args.dl_learner:
        args.single_task_baseline = True
    else:
        args.single_task_baseline = False

    if args.s_learner or args.single_task_baseline or args.sin_learner_stage1:
        args.task_embedding_config = 'early_concat'
        args.use_task_embeddings = False 

    if args.single_task_baseline and args.TEST_MODE == False:
        args.kill_seconds = 60

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(60*args.kill_seconds)  # Set a timer for 60 minutes (3600 seconds)


    ### Get the patients to exclude which have test/val drugs in their history
    exclusionLoader = QueryPatientsByDrug()

    df_val_test_split = pd.read_csv(args.val_test_location)
    test_tasks = list(df_val_test_split[df_val_test_split['split']=='test']['DrugBankID'])
    val_tasks = list(df_val_test_split[df_val_test_split['split']=='val']['DrugBankID'])


    if args.all_tasks_baseline or args.disable_exclusion or args.TEST_MODE:
        excludeTrainIDs = None
        excludeValIDs = None

    else:
        excludeTrainIDs = exclusionLoader(test_tasks+val_tasks)
        excludeValIDs = exclusionLoader(test_tasks)

        print("NEED TO HIDE PATIENTS BELOW FROM TRAIN ")
        print(excludeTrainIDs.shape)


    vocab_json = json.loads(open(VOCAB_JSON,'r').read())

    # when moving to lightning:
    #wandb_logger = WandbLogger(
    #    # name=job_name,
    #    project="dl-zero-shot-cate",
    #    entity="dsp",
    #    log_model=True,
    #    save_dir="outputs",
    #    settings=wandb.Settings(start_method="fork")
    #)

    # Device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device {device}')

    patientloader = PatientDataLoader(DATA_LOCATION)

    print('processing val data')

    valDatas = []
    counter = 0        

    if args.TEST_MODE:
        val_tasks = test_tasks
        splitStr = 'test'
    else:
        splitStr = 'val'

    for val_task in val_tasks:
        counter+=1
        #print('processing val data {}/{} for task {}'.format(counter,len(val_tasks),val_task))        
        valData = patientloader(val_task, split=splitStr, excludePatids=excludeValIDs,keep_treatment_col=((args.s_learner or args.single_task_baseline) and args.r_learner==False),include_support_in_val=(args.s_learner or args.single_task_baseline or args.all_tasks_baseline))
        if countPositiveLabels(valData) > args.min_pos_label:
            valDatas.append(valData)
            print('Loaded validation task {} which has {} positive labels'.format(val_task,countPositiveLabels(valData)))

    print('processing test data')
    #testData = patientloaderEval(test_task)


    # TODO: remove hard-coded paths to /home/anon and add relative paths to argparse:
    df_tasks = pd.read_csv(TASKS_CSV)
    df_tasks['order'] = [''.join(random.choice(ascii_letters) for x in range(5)) for _ in range(len(df_tasks))]

    if args.single_drugs_first:
        df_tasks['order_pure'] = df_tasks['order']
        df_tasks['order'] = df_tasks['ndrugs'].astype(str) + "__" + df_tasks['order']

    df_tasks = filter_for_used_drugs(df_tasks,
            used_drugs_path = 'task_embedding/tasks/used_drugs.csv')

    # TODO: filter out tasks that we don't have data for

    print("Number of tasks before filtering val/test {}".format(len(df_tasks)))
    for excludeTask in val_tasks + test_tasks:
        df_tasks = df_tasks[~((df_tasks['drug1']==excludeTask) | (df_tasks['drug2']==excludeTask))]
    print("Number of tasks before filtering val/test {}".format(len(df_tasks)))
    df_tasks = df_tasks[df_tasks['Drugs'].isin([x.split('tasks/')[1].replace('/','') for x in glob.glob(DATA_LOCATION+'tasks/*/',recursive=True)])]
    print("Number of tasks after filtering tasks without data {}".format(len(df_tasks)))
    df_tasks=df_tasks[df_tasks['ndrugs']<=args.max_train_drugs]
    print("Number of tasks after filtering out > {} drugs {}".format(args.max_train_drugs,len(df_tasks)))


    # Init mapping from drug to task embedding:
    te_mapper = TaskEmbeddingMapper(
        embedding_path = task_embedding_paths[args.kge],
        aggregation = args.task_embedding_aggregation,
        random = args.set_task_embeddings_to_random,
    )
    #add_path = lambda x: os.path.join(args.task_dir, x)
    #de_mapper = DrugEmbeddingMapper(
    #    task_path = add_path('tasks.pkl'),
    #    task_embedding_path = add_path('task_embeddings.pkl'),
    #    drug_path = add_path('used_drugs.csv'),
    #)

    task_embedding_config = args.task_embedding_config
    use_task_embeddings = args.use_task_embeddings

    input_dim = valDatas[0]['X'].shape[1]
    output_dim = 1
    task_dim = te_mapper('DB00563').shape[0]

    if args.sin_learner == False and args.graphite_learner == False:

        if use_task_embeddings:
            if task_embedding_config == 'early_concat':
                input_dim += task_dim
                print(f'For early_concat, using model input_dim: {input_dim}')
                # Initialize \PhiΦ, the initial parameter vector
                model = models.MLPModel(
                    input_dim,
                    output_dim,
                    args.dropout,
                    args.model_dim,
                    args.n_layers,
                    args.batch_norm,
                    layer_norm = False,
                    regress = (args.step == 'tau'),
                    residual = args.residual,
                    layer_norm_inner_layers = args.layer_norm_inner_layers,
                )
            elif task_embedding_config == 'late_concat':
                print(f'Using late_concat model')
                model = models.MLPLateConcatModel(
                    input_dim,
                    task_dim,
                    output_dim,
                    args.dropout,
                    args.model_dim,
                    args.n_layers,
                    args.batch_norm,
                    regress = (args.step == 'tau'),
                    residual = args.residual
                )
            elif task_embedding_config == 'late_concat_layernorm':
                print(f'Using late_concat_layernorm model')
                model = models.MLPLateConcatModelLayernorm(
                    input_dim,
                    task_dim,
                    output_dim,
                    args.dropout,
                    args.model_dim,
                    args.n_layers,
                    regress = (args.step == 'tau'),
                    residual = args.residual,
                    layer_norm_inner_layers = args.layer_norm_inner_layers,
                )
            elif task_embedding_config == 'hypernet':
                raise NotImplementedError()
            else:
                raise ValueError('No valid task_embedding_config provided')
        else:
            print('Using standard model w/o task embedding')
            # Initialize \PhiΦ, the initial parameter vector
            if task_embedding_config in ['late_concat', 'late_concat_layernorm']:
                raise ValueError('Late concat config requires use_task_embedddings=True')
            model = models.MLPModel(
                input_dim,
                output_dim,
                args.dropout,
                args.model_dim,
                args.n_layers,
                args.batch_norm,
                regress = (args.step == 'tau'),
                residual = args.residual
                )
    elif args.sin_learner == True:
        args.dim_node_features = task_dim
        args.model = 'sin'
        args.dim_covariates = input_dim
        from laboratory.sin_features import SIN
        model_m = torch.load(args.sin_learner_stage1_loc)
        model_m.to(device)
        args.como_net = model_m

        model = SIN(args)
        
    elif args.graphite_learner == True:
        args.dim_node_features = task_dim
        args.model = 'graphite'
        args.dim_covariates = input_dim
        from laboratory.graphite_features import GraphITE
        model = GraphITE(args)

    if args.model_location != '':
        model = torch.load(args.model_location)


    # TODO: getattr(models, model) init with model specific params
    #model = nn.DataParallel(model)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Training Model with {total_params} parameters.')
    print(model)

    criterion = nn.MSELoss()
    if args.loss == 'mae':
        criterion = nn.L1Loss()
    if args.loss == 'huber':
        criterion = nn.HuberLoss(delta=0.5)

    if args.s_learner:
        criterion = nn.BCELoss()

    #optimizer = get_optimizer(model, args.learning_rate, args.weight_decay)
    meta_optimizer = torch.optim.SGD(model.parameters(), lr=args.meta_learning_rate) #SGD
    inner_optimizer = get_optimizer(model, args.learning_rate, args.weight_decay) #AdamW
    inner_optimizer_state = inner_optimizer.state_dict() # get initial state

    n_iterations = args.n_iterations
    meta_batch_size = args.meta_batch_size

    torch.cuda.empty_cache()
    if args.use_lr_scheduler:
        raise NotImplementedError()
        print('Using LR scheduler')
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
                meta_optimizer,
                max_lr=args.learning_rate,
                steps_per_epoch=len(train_loader),
                epochs=n_iterations)


    # Init best value logger:
    # Pack all metrics to track best value in one dict:
    bv_log = BestValueLogger()

    # moved seeds here, as we don't want iterations to behave identical:
    torch.manual_seed(args.dummy_repetition)
    random.seed(args.dummy_repetition)
    np.random.seed(args.dummy_repetition)

    # Init the train adapter:
    train_adapter = camlAdapter(
        loss_fn = criterion,
        adaptation_steps = args.caml_k,
        eval_steps = 5,
        device = device,
        args = args
    )

    meta_lr = args.meta_learning_rate
    meta_lr_final = meta_lr * args.meta_learning_rate_final_ratio

    tasks_observed = []

    df_tasks.sort_values(by='order',inplace=True)
    df_tasks_cache = df_tasks.copy()

    if args.dry_load:
        print("Dry running task loader")
        count = 0 
        for ix, row in df_tasks.iterrows():
            TASK_DRUG = row['Drugs']
            print('\n\n\n LOADING TASK {} {}/{}\n\n\n'.format(TASK_DRUG,count,len(df_tasks)))
            count+=1
                
            taskdata = patientloader(TASK_DRUG, excludePatids=excludeTrainIDs,keep_treatment_col=(args.s_learner or args.single_task_baseline),include_support_in_val=(args.s_learner or args.single_task_baseline or args.all_tasks_baseline))
            m0 = RandomForestClassifier(random_state=0, n_estimators=150, max_depth=30,n_jobs=75,verbose=True)
            m0gen = XLearnerM0(m0)
            taskdata = m0gen(taskdata) # updated data with estimated tau label
            



    ####
    #### TODO: Skip training if T-learner, X-learner, X-trt Learner, R-learner 
    ####

    for iteration in range(n_iterations):
        print(f'>>>>>> Iteration {iteration}')

        if args.single_task_baseline == False and args.skip_training==False:
    
            # Each iteration, we train on <meta_batch_size> many tasks
    
            meta_optimizer.zero_grad() # reset meta optimizer
    
             # anneal meta-lr
            frac_done = float(iteration) / n_iterations
            new_lr = frac_done * meta_lr_final + (1 - frac_done) * meta_lr
            for pg in meta_optimizer.param_groups:
                pg['lr'] = new_lr
    
            # zero-grad the parameters
            for p in model.parameters(): # following l2l example
                p.grad = torch.zeros_like(p.data)
    
            meta_train_loss = 0.0
    
            for task in range(meta_batch_size):
                # Sample a task and load & preprocess corresponding data
                sample_weights = None
    
                if args.task_sample_weighting:
                    sample_weights = df_tasks['CNT']
    
                elif args.task_sample_weighting == 'log':
                    sample_weights = np.log(df_tasks['CNT'])
    
    
                if args.all_tasks_baseline and random.random() < args.all_tasks_baseline_eval_prob:
                    sampled = df_val_test_split.sample(n=1,weights=sample_weights).squeeze()
                    TASK_DRUG = sampled['Drugs']
                    tasks_observed.append(TASK_DRUG)
                    print("Sampled VAL/TEST task with drug: {}".format(TASK_DRUG))
                    # Load patient data for current drug / task:

                    if sampled['split'] == 'val':
                        excludeLocal = excludeValIDs
                    else:
                        excludeLocal = None


                    taskdata = patientloader(TASK_DRUG, excludePatids=excludeLocal,keep_treatment_col=args.s_learner,include_support_in_val=(args.s_learner or args.all_tasks_baseline),split=sampled['split'])
    
                else:
                    if args.deterministic_sample:
                        sampled = df_tasks.head(n=1).squeeze()
                        df_tasks=df_tasks[df_tasks['Drugs'] !=  sampled['Drugs']]
                        df_tasks.sort_values(by='order',inplace=True)
                        if len(df_tasks) == 0:
                            df_tasks = df_tasks_cache

                        if args.single_drugs_first and '_' in sampled['Drugs']:
                            df_tasks = df_tasks_cache
                            df_tasks['order'] = df_tasks['order_pure']
                            df_tasks.sort_values(by='order',inplace=True)
                            args.single_drugs_first = False

                    else:
                        sampled = df_tasks.sample(n=1,weights=sample_weights).squeeze()
                    ### TODO: Add back weighted sampling
                    #
                    TASK_DRUG = sampled['Drugs']
                    
                    tasks_observed.append(TASK_DRUG)
                    print("Sampled task with drug: {}".format(TASK_DRUG))
                    # Load patient data for current drug / task:
                    taskdata = patientloader(TASK_DRUG, excludePatids=excludeTrainIDs,keep_treatment_col=args.s_learner,include_support_in_val=(args.s_learner or args.all_tasks_baseline))
    
                if args.s_learner == False and args.all_tasks_baseline ==False:
                    # Train Y0 model and substract from tau
                    m0 = RandomForestClassifier(random_state=0, n_estimators=150, max_depth=30,n_jobs=75,verbose=True)
                    m0gen = XLearnerM0(m0)
                    taskdata = m0gen(taskdata) # updated data with estimated tau label    

                # TODO: move all of this out here, ideally just 1 liner to map taskdata to X_train_1, ..
                subsets = taskdata['subsets']
                
                subsets['train']['X'] = subsets['train']['X'].astype(np.float32)
                subsets['val']['X'] = subsets['val']['X'].astype(np.float32)
                subsets['test']['X'] = subsets['test']['X'].astype(np.float32)
    
                X_train = subsets['train']['X']
                X_val = subsets['val']['X']
                X_test = subsets['test']['X']
    
                y_train = subsets['train']['y']
                y_val = subsets['val']['y']
                y_test = subsets['test']['y']

                fractionPositive = 0
    
                if args.s_learner == True or args.sin_learner_stage1 == True:
                    input_train = X_train
                    input_val = X_val
                    input_test = X_test
    
                    target_train = torch.tensor(y_train.astype(np.float32))
                    target_val = torch.tensor(y_val.astype(np.float32))
                    target_test = torch.tensor(y_test.astype(np.float32))

                elif args.sin_learner == True or args.graphite_learner == True or args.s_learner_h:

                    input_train = X_train
                    input_val = X_val
                    input_test = X_test

    
                    target_train = torch.tensor(y_train.astype(np.float32))
                    target_val = torch.tensor(y_val.astype(np.float32))
                    target_test = torch.tensor(y_test.astype(np.float32))

                    train_treatment = subsets['train']['W']

                    if X_train.shape[0] > 512000:
                        downsample = np.random.choice(X_train.shape[0], size=512000, replace=False)
                        input_train = input_train[downsample]
                        target_train = target_train[downsample]
                        train_treatment = train_treatment[downsample]

                    val_treatment = subsets['val']['W']
                    test_treatment = subsets['test']['W']

                else:
                    tau_train = subsets['train']['tau']
                    tau_val = subsets['val']['tau']
                    tau_test = subsets['test']['tau']
    
                    train_treatment = subsets['train']['W']
                    val_treatment = subsets['val']['W']
                    test_treatment = subsets['test']['W']
        
                    ### ONLY THE TREATMENT PATIENTS
                    input_train = X_train[ subsets['train']['W'].nonzero()[0],:]
                    input_val = X_val[subsets['val']['W'].nonzero()[0],:]
                    input_test = X_test[subsets['test']['W'].nonzero()[0],:]
        
                    target_train = torch.tensor(tau_train[ subsets['train']['W'].nonzero()[0]].astype(np.float32))
                    target_val = torch.tensor(tau_val[subsets['val']['W'].nonzero()[0]].astype(np.float32))
                    target_test = torch.tensor(tau_test[subsets['test']['W'].nonzero()[0]].astype(np.float32))

                    fractionPositive = target_train.mean()

                if args.sin_learner == False and args.graphite_learner == False and args.s_learner_h == False:
                    if use_task_embeddings and args.s_learner_h == False:
                        print(f'Using task embeddings..')
                        # get task embedding:
                        task_embedding = te_mapper(TASK_DRUG)
                        # Dataloader expects additional empty dim:
                        task_embedding = task_embedding[None, :]
                        print(f' Using task embedding configuration: {task_embedding_config}')
                        do_concat = task_embedding_config == 'early_concat'
                        train_dataset = ClaimsTaskDataset(input_train, task_embedding, do_concat, target_train)
                        val_dataset = ClaimsTaskDataset(input_val, task_embedding, do_concat, target_val)
                    else:
                        print(f'Not using task embeddings.')
                        train_dataset = ClaimsDataset(input_train, target_train)
                        val_dataset = ClaimsDataset(input_val, target_val)
                        
                elif args.sin_learner == True or args.graphite_learner == True or args.s_learner_h == True:
                    task_embedding = te_mapper(TASK_DRUG)
                    null_task_embedding = te_mapper('NULL')

                    numRowsEmbedding = input_train.shape[0]
                    numRowsEmbeddingVal = input_val.shape[0]
                    treatmentEmbeddingDim = task_embedding.shape[0]

                    treatment_matrix = np.broadcast_to(train_treatment, (treatmentEmbeddingDim,numRowsEmbedding)).T
                    y1_t_matrix = np.broadcast_to(task_embedding,(numRowsEmbedding,treatmentEmbeddingDim))
                    y0_t_matrix = np.broadcast_to(null_task_embedding,(numRowsEmbedding,treatmentEmbeddingDim))

                    treatment_features_matrix = np.multiply(treatment_matrix,y1_t_matrix) + np.multiply(1-treatment_matrix,y0_t_matrix)
                    train_dataset = SINTaskDataset(input_train, treatment_features_matrix, target_train)

                    treatment_matrix = np.broadcast_to(val_treatment, (treatmentEmbeddingDim,numRowsEmbeddingVal)).T
                    y1_t_matrix = np.broadcast_to(task_embedding,(numRowsEmbeddingVal,treatmentEmbeddingDim))
                    y0_t_matrix = np.broadcast_to(null_task_embedding,(numRowsEmbeddingVal,treatmentEmbeddingDim))
                    val_dataset = SINTaskDataset(input_val, treatment_features_matrix, target_val)



                #added Random Sampler to the train_loader, 
                # train_loader will now loop for args.caml_k steps, 
                # and in each step args.batch_size many points will be sampled from train_dataset
                if args.sin_learner == False:

                    sampler = RandomSampler(train_dataset,replacement=True,num_samples=args.caml_k*args.batch_size)
                    train_loader = DataLoader(
                            train_dataset, 
                            batch_size = args.batch_size, 
                            num_workers=0,
                            collate_fn=sparse_batch_collate if (args.s_learner_h == False and args.graphite_learner==False) else sparse_batch_collate3, 
                            sampler=sampler if args.use_random_sampler else None,
                            shuffle=True
                    )
                    val_loader = DataLoader(val_dataset, batch_size = args.batch_size, num_workers=0,  collate_fn=sparse_batch_collate if (args.s_learner_h == False and args.graphite_learner==False) else sparse_batch_collate3 )
        
                    learner = copy.deepcopy(model) # model to adapt
                    # init new optimizer for task
                    inner_optimizer = get_optimizer(learner, args.learning_rate, args.weight_decay)
                    inner_optimizer.load_state_dict(inner_optimizer_state)
        
                    # Adapt model to current task:
                    inner_val_loss = train_adapter(
                        train_loader,
                        val_loader,
                        learner,
                        inner_optimizer,
                    )
        
                    # Get new optimizer state:
                    inner_optimizer_state = inner_optimizer.state_dict()
                    # Update gradient of outer model for SGD: Add Phi tilde component
                    for p, l in zip(model.parameters(), learner.parameters()):
                        p.grad.data.add_(-1.0, l.data )
                    meta_train_loss += inner_val_loss.item()

                elif args.sin_learner == True:

                    sampler = RandomSampler(train_dataset,replacement=True,num_samples=args.caml_k*args.batch_size)

                    train_loader = DataLoader(
                            train_dataset, 
                            batch_size = args.batch_size, 
                            num_workers=0,
                            collate_fn=sparse_batch_collate if (args.s_learner_h == False and args.graphite_learner==False and args.sin_learner == False) else sparse_batch_collate3, 
                            sampler=sampler if args.use_random_sampler else None,
                            shuffle=True
                    )
                    #val_loader = DataLoader(val_dataset, batch_size = args.batch_size, num_workers=0,  collate_fn=sparse_batch_collate if (args.s_learner_h == False and args.graphite_learner==False) else sparse_batch_collate3 )
        
                    learner = copy.deepcopy(model) # model to adapt
                    inner_optimizer = get_optimizer(learner, args.learning_rate, args.weight_decay)
                    inner_optimizer.load_state_dict(inner_optimizer_state)

                    optimizer_dict = {}
                    optimizer_dict['covariates_net_opt'] = get_optimizer_scheduler(args=args, model=model.covariates_net)[0]
                    optimizer_dict['treatment_net_opt'] =  get_optimizer_scheduler(args=args, model=model.treatment_net, net="gnn")[0]
                    optimizer_dict['propensity_net_opt'] = get_optimizer_scheduler(args=args, model=model.propensity_net, net="propensity")[0]

                    # Adapt model to current task:
                    inner_val_loss = train_adapter(
                        train_loader,
                        None,
                        learner,
                        None,
                        optimizer_dict
                    )
        
                    # Get new optimizer state:
                    inner_optimizer_state = inner_optimizer.state_dict()
                    # Update gradient of outer model for SGD: Add Phi tilde component
                    for p, l in zip(model.parameters(), learner.parameters()):
                        p.grad.data.add_(-1.0, l.data )
                    meta_train_loss += inner_val_loss.item()






                    # inner_val_loss = 0
# 
                    # train_loader = DataLoader(
                    #     train_dataset, 
                    #     batch_size = args.batch_size, 
                    #     num_workers=0,
                    #     sampler=None,
                    #     collate_fn=sparse_batch_collate3,
                    #     shuffle=True
                    # )
# 
                    # model.train_step(device, train_loader, epoch=iteration, log_interval=1, max_steps=args.caml_k)

    
    
                logs = {"iteration": iteration, "tasks_observed":len(tasks_observed), "tasks_observed_unique":len(set(tasks_observed)), "task": task, "inner_val_loss": inner_val_loss, "task_drug": TASK_DRUG}
                
                file = open(os.path.join(getWANDBDir(run), "iterlog.out"),'a')
                file.write('{},{},{},{},{},{},{},{},{}\n'.format(iteration,str(TASK_DRUG),inner_val_loss, meta_train_loss, task,len(tasks_observed),fractionPositive,countPositiveLabels(taskdata),input_train.shape[0] ))
                file.close()

                wandb.log(logs)
    
    
                del taskdata
                del train_loader
                if args.sin_learner == False and args.graphite_learner == False:
                    del val_loader
                    del val_dataset
                del train_dataset
                
    
                # It should be as simple as:
                # train_loader, val_loader = get_task('validation') where train/val refer to
                # the splits within the task, and validation refers to held-out tasks
    
            # Average accumulated meta_train_loss by meta_batch_size
            meta_train_loss /= meta_batch_size 
            iter_log = {"iteration": iteration, "tasks_observed":len(tasks_observed), "tasks_observed_unique":len(set(tasks_observed)) ,"meta_train_loss": meta_train_loss, "meta_learning_rate": new_lr}
            


            # Update gradient of outer model: average phi tilde over tasks, add old phi
            for p in model.parameters():
                p.grad.data.mul_(1.0 / meta_batch_size).add_(p.data)
            meta_optimizer.step()
            meta_optimizer.zero_grad()

        #
        # TODO: encapsulate this into 2-3 lines of code
        #
        #
        if iteration%args.val_interval==0 or args.TEST_MODE == True:

            if args.single_task_baseline == True or args.skip_training == True:
                iter_log = {}
            ### For each of the zero-shot validation tasks

            if args.TEST_MODE:
                num_treatments_all = []
                num_positive_all = []
                test_tasks_all = []

            for valData in valDatas:




                ####
                #### TODO: Train and evaluate for T-learner, X-learner, X-trt Learner, R-learner 
                ####


                # Extract the task name, patient features (X_test)
                valTask = valData['task']
                subsets = valData['subsets']
                X_test = subsets['test']['X'].astype(np.float32)
                y_val = torch.tensor(subsets['test']['y'].astype(np.float32))
                val_treatment = subsets['test']['W']

                if args.TEST_MODE:
                    iter_log = {}
                    print("Testing for task {}".format(valData['task']))

                    num_treatments = float(subsets['test']['W'].sum())
                    num_positive = float(y_val[subsets['test']['W'].nonzero()[0]].sum())
            
                    if num_positive < 5:
                        print("Skipping task {} because only {} positive samples".format(test_task, num_positive))
                        continue
            
                    num_treatments_all.append(num_treatments)
                    num_positive_all.append(num_positive)
                    test_tasks_all.append(valTask)



                if args.sin_learner == False and args.graphite_learner == False and args.s_learner_h == False:
                    if args.single_task_baseline == False:
    
                        if use_task_embeddings:
                            # Load the drug embedding for the task
                            task_embedding = te_mapper(valTask)
                            task_embedding = task_embedding[None, :]
        
                            do_concat = task_embedding_config == 'early_concat'
                            meta_val_dataset = ClaimsTaskDataset(X_test, task_embedding, do_concat, y_val)
                        else:
        
                            if args.s_learner == False:
        
                                meta_val_dataset = ClaimsDataset(X_test, y_val)
                            else:
                                trt_columns = [vocab_json['Drugs'].index(x) for x in valTask.split("_")]
                                for trt_column in trt_columns:
                                    X_test[:,trt_column] = 1
        
                                X_test_0 = X_test.copy()
        
                                for trt_column in trt_columns:
                                    X_test_0[:,trt_column] = 0
        
                                meta_val_dataset = ClaimsSLearnerDataset(X_test, X_test_0, y_val)
        
                        if args.s_learner == False:
                            meta_val_loader = DataLoader(meta_val_dataset, batch_size = args.batch_size, num_workers=0,  collate_fn=sparse_batch_collate)
                        else:
                            meta_val_loader = DataLoader(meta_val_dataset, batch_size = args.batch_size, num_workers=0,  collate_fn=sparse_batch_collate2)
        
        
                        with torch.no_grad():
        
                            model.eval()
                            val_loss = 0.0
                            n_samples = 0
                            tau_preds = []
        
                            ## Loop through batch and make predictions
                            ## Note that if we are concatenating the drug embedding later (late_concat)
                            # Then we pass it into the model separately here
                            for eval_iter, batch in enumerate(tqdm(meta_val_loader)):
                                if args.task_embedding_config in ['late_concat', 'late_concat_layernorm']:
                                    data, task_embedding, labels = batch
                                    data  = data.to(device)
                                    task_embedding = task_embedding.to(device)
                                    tau_pred = model(data, task_embedding)
                                    task_embedding.detach()
        
                                else:
                                    if args.s_learner == False: 
                                        data, labels = batch
                                        data = data.to(device)
                                        tau_pred = model(data)
                                    else:
                                        data,data2, labels = batch
                                        data = data.to(device)
                                        data2 = data2.to(device)       
                                        tau_pred = model(data) - model(data2)                         
        
                                tau_preds.append(tau_pred.view(-1))
        
                            meta_tau_pred = torch.cat(tau_preds)
    
                    elif args.single_task_baseline == True:
                        meta_learner = None
    
                        if args.dl_learner == False:
    
                            if args.x_trt_learner:
                                meta_learner = XTrtLearner(args=args)
                            if args.t_learner:
                                meta_learner = TLearner(args=args)
                            if args.x_learner:
                                meta_learner = XLearner(args=args)
                            if args.r_learner:
                                meta_learner = RLearner(args=args)
        
                            meta_learner.train(subsets['train'])
                            meta_tau_pred = torch.tensor(meta_learner.predict(X_test))
    
                        else:
    

                            from catenets.models.torch.representation_nets import TARNet, DragonNet
                            from catenets.models.torch.flextenet import FlexTENet

                            
                            if args.dl_learner_type == 'TARNet':
                                meta_learner = TARNet(n_unit_in=input_dim,n_iter_print=1,batch_size=args.batch_size,n_iter=args.dl_n_iter, n_layers_r=args.dl_n_layers_r,n_units_r=args.dl_n_units_r,n_layers_out=args.dl_n_layers_out,n_units_out=args.dl_n_units_out,weight_decay=args.dl_weight_decay,lr=args.dl_lr,dropout=args.dl_dropout , penalty_disc=args.dl_penalty_disc)
    
                            if args.dl_learner_type == 'DragonNet':
                                meta_learner = DragonNet(n_unit_in=input_dim,n_iter_print=1,batch_size=args.batch_size,n_iter=args.dl_n_iter, n_layers_r=args.dl_n_layers_r,n_units_r=args.dl_n_units_r,n_layers_out=args.dl_n_layers_out,n_units_out=args.dl_n_units_out,weight_decay=args.dl_weight_decay,lr=args.dl_lr,dropout=args.dl_dropout  )
            
                            if args.dl_learner_type == 'FlexTENet':
                                meta_learner = FlexTENet(binary_y=True,n_unit_in=input_dim,n_iter_print=1,batch_size=args.batch_size,n_iter=args.dl_n_iter, n_layers_r=args.dl_n_layers_r,n_layers_out=args.dl_n_layers_out,weight_decay=args.dl_weight_decay,lr=args.dl_lr,dropout=args.dl_dropout ,   n_units_s_out=args.dl_n_units_s_out, n_units_p_out=args.dl_n_units_p_out, n_units_s_r=args.dl_n_units_s_r,n_units_p_r=args.dl_n_units_p_r, private_out=args.dl_private_out, penalty_orthogonal=args.dl_penalty_orthogonal)
        
                            meta_learner.fit(subsets['train']['X'], subsets['train']['y'], subsets['train']['W'])
                            
                            numRowsEmbedding = X_test.shape[0]

                            tau_preds = []
                            for i in range(0,20):
                                start_i = i*50000
                                print(start_i)
                                end_i = min(numRowsEmbedding, (i+1)*50000)


                                tau_pred = torch.tensor(meta_learner.predict(X_test[start_i:end_i,:])).to(device)
                                tau_preds.append(tau_pred.view(-1))

                                if end_i == numRowsEmbedding:
                                    break

                            meta_tau_pred = torch.cat(tau_preds)




                elif args.sin_learner == True or args.graphite_learner == True or args.s_learner_h == True:
                    task_embedding = te_mapper(valTask)
                    null_task_embedding = te_mapper('NULL')
    
                    numRowsEmbedding = X_test.shape[0]
                    treatmentEmbeddingDim = task_embedding.shape[0]
    
                    treatment_matrix = np.broadcast_to(val_treatment, (treatmentEmbeddingDim,numRowsEmbedding)).T
                    y1_t_matrix = np.broadcast_to(task_embedding,(numRowsEmbedding,treatmentEmbeddingDim))
                    y0_t_matrix = np.broadcast_to(null_task_embedding,(numRowsEmbedding,treatmentEmbeddingDim))
    
                    treatment_features_matrix_test = np.multiply(treatment_matrix,y1_t_matrix) + np.multiply(1-treatment_matrix,y0_t_matrix)

                    covariates_test, treatment_node_features_test, target_outcome_test = X_test,torch.Tensor(treatment_features_matrix_test),y_val

                    tau_preds = []

                    for i in range(0,20):
                        start_i = i*50000
                        print(start_i)
                        end_i = min(numRowsEmbedding, (i+1)*50000)


                        covariates_test_subset  = sparse_csr_to_tensor(covariates_test[start_i:end_i,:]).to(device)
                        treatment_node_features_test_subset  = treatment_node_features_test[start_i:end_i,:].to(device)
                        target_outcome_test_subset  = target_outcome_test[start_i:end_i].to(device)
    
    
                        treatment_node_features_test_y1  = torch.Tensor(y1_t_matrix[start_i:end_i,:]).to(device)
                        treatment_node_features_test_y0  = torch.Tensor(y0_t_matrix[start_i:end_i,:]).to(device)
    
    
                        if args.sin_learner == True:
                            allTestData = (covariates_test_subset, treatment_node_features_test_subset, target_outcome_test_subset)
        
                            with torch.no_grad():
                                tau_pred = torch.tensor(model.test_prediction(allTestData))
                                tau_preds.append(tau_pred.view(-1))
    
                        elif args.graphite_learner == True : 
                            allTestData_y1 = (covariates_test_subset, treatment_node_features_test_y1, target_outcome_test_subset)
                            allTestData_y0 = (covariates_test_subset, treatment_node_features_test_y0, target_outcome_test_subset)
    
                            with torch.no_grad():
                                tau_pred = torch.tensor(model.test_prediction(allTestData_y1)) -  torch.tensor(model.test_prediction(allTestData_y0)) 
                                tau_preds.append(tau_pred.view(-1))
    
                        elif args.s_learner_h == True:
    
                            with torch.no_grad():

                                tau_pred = torch.tensor(model(covariates_test_subset, treatment_node_features_test_y1)) -  torch.tensor(model(covariates_test_subset, treatment_node_features_test_y0)) 
                                tau_preds.append(tau_pred.view(-1))

                        if end_i == numRowsEmbedding:
                            break

                    meta_tau_pred = torch.cat(tau_preds)


                recall_thresholds = [0.001,0.002,0.005,0.01]
                rate_thresholds = [0.001,0.002,0.005,0.01]

                # Compute eval metrics (recall, rate)
                table_recall = treatmentRecall(
                        subsets['test'],
                        list(meta_tau_pred.cpu().numpy()),
                        recall_thresholds
                )
                table_rate = treatmentRATE(
                        subsets['test'],
                        list(meta_tau_pred.cpu().numpy()),
                        RandomForestClassifier(
                            random_state=0,
                            n_estimators=150,
                            max_depth=30,
                            n_jobs=75,
                            verbose=True),
                        rate_thresholds
                )
                table_precision = treatmentPrecision(
                    subsets['test'], 
                    list(meta_tau_pred.cpu().numpy()),
                    recall_thresholds 
                )
                table_choose2 = treatmentChoose2(subsets['test'], meta_tau_pred.cpu().numpy())
                # the first time, we collect the metric names for best value logger
                addLogDict(iter_log, table_recall,valTask,split='Val',metric_name='recall',higherIsBetter=True)
                addLogDict(iter_log, table_rate,valTask,split='Val',metric_name='rate',higherIsBetter=True)
                if args.TEST_MODE == True:
                    addLogDict(iter_log, table_precision,valTask,split='Test',metric_name='precision')
                    addLogDict(iter_log, table_choose2, valTask, split='Test',metric_name='choose2')
                    performance_logs[(args.output_suffix,valTask)] = iter_log
                
                    performance_log_loc = MODEL_OUTPUTS+ 'performance_logs/'
                    os.makedirs(performance_log_loc,exist_ok=True)
                    performance_logs_final = logsToTable(performance_logs).merge(pd.DataFrame({'task':  test_tasks_all,'numPositive':num_positive_all, 'numTreatments':num_treatments_all}),on='task')
                    performance_logs_final.to_csv(performance_log_loc+"__"+args.val_test_location.split('/')[-1]+"__"+args.output_suffix)
                    print(performance_logs_final)


                if args.TEST_MODE == False:
                    torch.save(model, os.path.join(getWANDBDir(run), "model_{}.pt".format(iteration)))

                try:
                    del meta_val_dataset
                    del meta_val_loader
                    del data
                    if args.s_learner:
                        del data2
                    del labels
                    del tau_preds
                    del meta_tau_pred

                    if args.sin_learner or args.graphite_learner:
                        del covariates_test
                        del treatment_node_features_test
                        if args.graphite_learner:
                            del treatment_node_features_test_y1
                            del treatment_node_features_test_y0
                        del target_outcome_test

                except:
                    pass


            ####
            #### All metrics in iter_log contain either ↑ (higher is better) or ↓ (lower is better)
            #### * Log best values
            #### * Log average value
            ####
            # extract metrics for best value logger
            result_dict = {}

            if args.TEST_MODE == False:

                for metricName, metricValue in iter_log.items():
    
                    print(metricName, metricValue)
    
                    if '↑' not in metricName and '↓' not in metricName:
                        continue
    
                    print(metricName, metricValue)
                    name_split = metricName.split(' ')
                    metric = name_split[1] # hard coded due to nasty format of metricName
                    thres = name_split[3]
                    curr_task = name_split[4].lstrip('(').rstrip(')')
                    if thres not in result_dict.keys():
                        result_dict[thres] = {}
                    if metric not in result_dict[thres].keys():
                        result_dict[thres][metric] = {}
                    # populate dict:
                    result_dict[thres][metric][curr_task] = metricValue
                    if float(thres) == 0.998: #only log single thres for all drugs
                        bv_log(
                            name = metricName,
                            value = metricValue,
                            step = iteration,
                            increasing = False
                        )
                # log averages:
                for thres in result_dict.keys():
                    for metric in result_dict[thres].keys():
                        mean_value = np.mean([ result_dict[thres][metric][task]
                            for task in result_dict[thres][metric].keys()
                        ])
                        mean_metric_name = ' '.join(['Mean Val', metric, '@', thres])
                        bv_log(
                            name = mean_metric_name,
                            value = mean_value,
                            step = iteration,
                            increasing = False #hard-coded for current metrics
                        )
                        iter_log.update({mean_metric_name: mean_value})
    
                # log best values
                wandb.log(
                    bv_log.best_values
                )

        if args.TEST_MODE == False:
            wandb.log(
                iter_log
            )

        if args.single_task_baseline or args.TEST_MODE:
            break

    #Perform k &gt; 1k>1 steps of SGD on task TT, starting with parameters \PhiΦ, resulting in parameters WW
    #Update: \Phi \gets \Phi + \epsilon (W-\Phi)Φ←Φ+ϵ(W−Φ)



    # Experiments:
    # Compare performance of zero shot to
        # Random
        # X treat (random forest)
        # X treat (neural network)


def print_memory():
    x = torch.cuda.memory_allocated()
    print(f'CUDA MEMORY USAGE : {x/(10**9)} GB')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_interval', type=int,
                        default=5,
                        help='Number of intervals before val')
    parser.add_argument('--step', choices=['tau','y0'], type=str, default='tau')
    parser.add_argument('--dataset', default='DB00563-Pancytopenia')
    # Training config:
    parser.add_argument('--n_iterations', type=int,
                        default=200,
                        help='Number of meta-batches that are trained over')
    parser.add_argument('--learning_rate', type=float,
                        default=0.001)
    parser.add_argument('--meta_learning_rate', type=float,
                        default=1) # =1 in l2l example
    parser.add_argument('--meta_learning_rate_final_ratio', type=float,
                        default=0.1, help='Ratio between final and start meta lr.') # =1 in l2l example
    parser.add_argument('--batch_size', type=int,
                        default=2048*4)
    parser.add_argument('--meta_batch_size', type=int,
                        default=4)
    parser.add_argument('--l1_reg', type=float,
                        default=5e-7) # of the encoder layer
    parser.add_argument('--use_lr_scheduler', type=str,
                        default="false") # hack to cope with wandb
    # Model params:
    parser.add_argument('--n_layers', type=int,
                        default=6)
    parser.add_argument('--model_dim', type=int,
                        default=256)
    parser.add_argument('--dry_load', type=str, default="false") # hack to cope with wandb
    parser.add_argument('--batch_norm', type=str, default="true") # hack to cope with wandb
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--loss', choices=['mse','huber','mae'], type=str,
                        default='mse')
    parser.add_argument('--caml_k', type=int, default=50)
    parser.add_argument('--val_test_location', type=str,  default='metalearn/splits/icml_new_single_split.csv')
    #parser.add_argument('--task_dir', type=str, default='task_embedding/tasks/',
    #                    help='reative path to task infos (embeddings, drugs) from repo root')
    parser.add_argument('--use_task_embeddings', type=str, default="true") # hack to cope with wandb
    parser.add_argument('--set_task_embeddings_to_random', type=int, default=0)
    parser.add_argument('--task_embedding_config', type=str, default="late_concat",
                        choices=['early_concat', 'late_concat', 'late_concat_layernorm'], help='How to include task embeddings')
    parser.add_argument('--residual', type=str, default="true") # hack to cope with wandb
    parser.add_argument('--kge', type=str, default="stargraph",
                        help='What KG embedding to use [distmult, stargraph]')
    parser.add_argument('--task_embedding_aggregation', type=str, default="sum",
                        help='Aggregation function for task embeddings (multiple drugs) [sum, mean]')
    parser.add_argument('--task_sample_weighting', type=str, default="disable", choices =['disable','enable','log'],
                        help='Weight tasks during sampling')
    parser.add_argument('--model_location', type=str,  default='')
    parser.add_argument('--max_train_drugs', type=int,  default=2)
    parser.add_argument('--min_pos_label', type=int,  default=50)
    parser.add_argument('--dummy_repetition', type=int,  default=1,
            help="""otherwise inactive repetition number to
                    more easily start repetitions with the same config in wandb""")
    parser.add_argument('--use_random_sampler', type=str, default="false") 
    parser.add_argument('--layer_norm_inner_layers', type=str, default="false") 


    parser.add_argument('--all_tasks_baseline', type=str, default="false") 
    parser.add_argument('--all_tasks_baseline_eval_prob', type=float, default=0.01) 
    parser.add_argument('--zero_shot_baseline', type=str, default="false") 

    parser.add_argument('--s_learner_h', type=str, default="false") 
    parser.add_argument('--s_learner', type=str, default="false") 
    parser.add_argument('--x_trt_learner', type=str, default="false") 
    parser.add_argument('--x_learner', type=str, default="false") 
    parser.add_argument('--t_learner', type=str, default="false") 
    parser.add_argument('--r_learner', type=str, default="false") 
    parser.add_argument('--dl_learner', type=str, default="false") 
    parser.add_argument('--dl_learner_type', type=str, default="TARNet", choices = ['TARNet','DragonNet','FlexTENet']) 
    parser.add_argument('--sin_learner', type=str, default="false") 
    parser.add_argument('--sin_learner_stage1', type=str, default="false") 
    parser.add_argument('--graphite_learner', type=str, default="false") 

    parser.add_argument('--rf_n_estimators', type=int, default=150) 
    parser.add_argument('--rf_max_depth', type=int, default=30) 
    parser.add_argument('--rf_min_samples_split', type=int, default=2) 
    parser.add_argument('--rf_criterion_regress', type=str, default="squared_error") 
    parser.add_argument('--rf_criterion_binary', type=str, default="gini") 
    parser.add_argument('--rf_max_features', type=str, default="sqrt") 

    parser.add_argument('--dl_n_layers_r', type=int, default=3) 
    parser.add_argument('--dl_n_units_r', type=int, default=200) 
    parser.add_argument('--dl_n_layers_out', type=int, default=2) 
    parser.add_argument('--dl_n_units_out', type=int, default=100) 
    parser.add_argument('--dl_weight_decay', type=float, default=1e-4) 
    parser.add_argument('--dl_lr', type=float, default=0.0001 ) 
    parser.add_argument('--dl_n_iter', type=int, default=10000) 
    parser.add_argument('--dl_n_batch_size', type=int, default=8096) 
    parser.add_argument('--dl_dropout', type=str, default='false') 
    parser.add_argument('--dl_penalty_disc', type=float, default=1e-4) 

    parser.add_argument('--dl_n_units_s_out', type=int, default=64) 
    parser.add_argument('--dl_n_units_p_out', type=int, default=64) 
    parser.add_argument('--dl_n_units_s_r', type=int, default=128) 
    parser.add_argument('--dl_n_units_p_r', type=int, default=128) 
    parser.add_argument('--dl_private_out', type=str, default='false') 
    parser.add_argument('--dl_penalty_orthogonal', type=float, default=0.001) 


    parser.add_argument('--disable_exclusion', type=str, default="false") 
    parser.add_argument('--deterministic_sample', type=str, default="true") 
    parser.add_argument('--single_drugs_first', type=str, default="true") 

    parser.add_argument('--skip_training', type=str, default="false") 


    # all related to the SIN model
    parser.add_argument('--sin_learner_stage1_loc', type=str, default='No sin m model location provided') 
    parser.add_argument('--dim_output', type=int, default=256) 
    parser.add_argument('--dim_hidden_treatment', type=int, default=256) 
    parser.add_argument('--num_treatment_layer', type=int, default=4) 
    # parser.add_argument('--mlp_batch_norm', type=int, default=256) 
    parser.add_argument('--initialiser', type=str, default='xavier') 
    parser.add_argument('--activation', type=str, default='relu') 
    parser.add_argument('--leaky_relu', type=float, default=0.1) 
    parser.add_argument('--output_activation_treatment_features', type=str, default='true') 
    parser.add_argument('--dim_hidden_propensity', type=int, default=16) 
    parser.add_argument('--num_propensity_layers', type=int, default=3) 
    parser.add_argument('--pro_dropout', type=float, default=0.0) 
    parser.add_argument('--dim_hidden_como', type=int, default=256) 
    parser.add_argument('--num_como_layers', type=int, default=3) 
    parser.add_argument('--como_dropout', type=float, default=0.0) 
    parser.add_argument('--sin_weight_decay', type=float, default=0.0) 
    parser.add_argument('--lr', type=float, default=1e-3) 
    parser.add_argument('--optimizer', type=str, default='sgd') 
    parser.add_argument('--lr_scheduler', type=str, default='none') 
    parser.add_argument('--dim_hidden_covariates', type=int, default=128) 
    parser.add_argument('--num_covariates_layer', type=int, default=3) 
    parser.add_argument('--como_lr', type=float, default=0.001) 
    parser.add_argument('--pro_weight_decay', type=float, default=0.000) 
    parser.add_argument('--gnn_weight_decay', type=float, default=0.000) 
    parser.add_argument('--pro_lr', type=float, default=0.001) 
    parser.add_argument('--num_update_steps_como', type=int, default=10) 
    parser.add_argument('--num_update_steps_propensity', type=int, default=10) 
    parser.add_argument('--num_update_steps_global_objective', type=int, default=10) 
    parser.add_argument('--log_interval', type=int, default=1) 


    parser.add_argument('--independence_regularisation_coeff', type=float, default=0.01) 
    parser.add_argument('--dim_output_treatment', type=int, default=256) 
    parser.add_argument('--dim_output_covariates', type=int, default=256) 
    parser.add_argument('--num_final_ff_layer', type=int, default=4) 

    parser.add_argument('--kill_seconds', type=int, default=7200) 


    parser.add_argument('--TEST_MODE', type=str, default='false') 
    parser.add_argument('--output_suffix', type=str, default="", help='Concatenate this to the end of the file')

    ###
    ### TODO: Add random forest parameters for T-learner, X-Learner, X-Trt-Learner, R-learner
    ###
    ###

    args = parser.parse_args()
    
    # update pseudo-bool arguments:
    args_to_convert = [
            key for key,val in vars(args).items() if val in ["true", "false"]
    ]  #['batch_norm', 'use_lr_scheduler', 'use_task_embeddings', 'residual', 'task_sample_weighting']
    args = convert_str_to_bool(args, args_to_convert)
    print(f'Boolean args: {args_to_convert}')

    # Setting the seeds:
    torch.manual_seed(args.dummy_repetition)
    np.random.seed(args.dummy_repetition)
    random.seed(args.dummy_repetition)

    args.mlp_batch_norm = args.batch_norm
    if args.sin_learner == True or args.graphite_learner == True:
        args.weight_decay = args.sin_weight_decay

    if args.TEST_MODE == True:
        args.skip_training = True

    main(args)



#print('Reached end of loop---- {}------'.format(task))
#print(torch.cuda.memory_allocated())
#print(torch.cuda.max_memory_allocated())
#print('Reached end of loop---- {}  --')
