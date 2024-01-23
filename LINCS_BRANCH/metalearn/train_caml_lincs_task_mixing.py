import sys
import os
sys.path.append(os.getcwd())

import gc
import argparse
import pdb
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
from sklearn.metrics import mean_squared_error
# Custom modules:
from causal_singletask.consts_and_utils import *
import src.models
from src import models
from causal_singletask.torch_utils import BestValueLogger, EarlyStopping
from metalearn.eval import treatmentRecall, treatmentPrecision, addLogDict, treatmentRATE, treatmentChoose2, countPositiveLabels
from metalearn.parse import CellDataLoader, XLearnerM0, QueryPatientsByDrug
from metalearn.caml import camlAdapter


from metalearn.bio_utils import generate_trainvaltest_metadata_split, mean_batch, TaskEmbeddingMapper, get_data_for_task, BioDataset, \
    BioTaskDataset, BioSINTaskDataset, get_index_for_most_diff_genes

from pdb import set_trace as bp
from causal_singletask.models.xlearner import (XTrtLearner, TLearner,XLearner, RLearner)

import signal
import time

import scanpy as sc


import importlib


def get_DE_gene_idx(adata, k):
    return np.array(np.argsort(np.abs(adata.X).mean(0))[::-1][:k]).tolist()


def handler(signum, frame):
    # This function will be called when the timer expires
    raise SystemExit("Script ran for more than 60 minutes, exiting.")


def train_y0_model(y_zero, X, test_split=False):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression

    cell_names = y_zero.index.values
    if test_split:
        train, test = train_test_split(cell_names, test_size=0.1, random_state=1)
    else:
        train = cell_names

    regressors = {}
    test_mses = {}
    print("Training Y(0) Model")

    for itr, gene in enumerate(y_zero.columns):
        colname = [x for x in X.columns if gene in x][0]
        y = y_zero.loc[train.astype('str'), gene].values

        X_train = X.loc[train.astype('str'), colname].values
        X_train = X_train.reshape(-1, 1)
        regressors[gene] = LinearRegression().fit(X_train, y)

        if test_split:
            X_test = X.loc[test.astype('str')].iloc[:, 1:].values
            X_test = X_test.reshape(-1, 1)

            y_hat_test = regressors[gene].predict(X_test)
            y_test = y_zero.loc[test.astype('str'), gene].values
            test_mses[gene] = np.mean((y_hat_test - y_test) ** 2)

    if test_split:
        print("Y_0 prediction test MSE:", np.mean(list(test_mses.values())))

    return regressors


def get_tau(cell_lines, saved_y0_models, y_zero, Y, W, feats, y0_flag='impute'):
    cell_lines = np.array(cell_lines)
    genes = y_zero.columns
    untreat_cell_lines = W[W.iloc[:,0]==0].index.values
    treat_cell_lines = W[W.iloc[:,0]==1].index.values

    if y0_flag == 'impute':
        key = str(untreat_cell_lines)
        try:
            B, c = saved_y0_models[key]
        except:
            gene_order = y_zero.columns.values.astype('str')
            y0_model = train_y0_model(y_zero.loc[untreat_cell_lines],
                feats.loc[untreat_cell_lines], test_split=False)
            B = [y0_model[gene].coef_[0] for gene in gene_order]
            c = [y0_model[gene].intercept_ for gene in gene_order]
            saved_y0_models[key] = (B, c)

        ## Impute for treated-train cell lines
        y0_treat = np.multiply(feats.loc[treat_cell_lines, genes].values, B)
        y0_treat = np.add(y0_treat, c)

    elif y0_flag == 'real':
        y0_treat = y_zero.loc[treat_cell_lines].values
    else:
        raise Exception('Wrong y0 flag')

    tau_treat = Y.loc[treat_cell_lines] - y0_treat

    ## TODO Update to include y1 imputation
    tau_untreat = pd.DataFrame(index=untreat_cell_lines,columns=tau_treat.columns)

    tau = pd.concat([tau_treat, tau_untreat])
    tau = tau.loc[cell_lines]
    return tau




def LINCSDataLoader(task, split, keep_treatment_col, y_zero,
                    include_support_in_val, adata, feats, saved_y0_models,
                    test_size=0.1, random_state=1):
    all_tasks = adata.obs['pert_id'].unique()
    split_adata = adata[adata.obs['split'] == split.title()]
    task_adata = split_adata[split_adata.obs['pert_id'] == task]
    if len(task_adata)==0:
        raise ValueError('Specified split doesn not match the task')
    all_cell_lines = split_adata.obs['cell_iname'].unique().tolist()
    cell_lines_treat = task_adata.obs['cell_iname'].unique().tolist()
    cell_lines_untreat = list(set(all_cell_lines).difference(set(
        cell_lines_treat)))

    y_one = task_adata.to_df()
    y_one['cell_iname'] = task_adata.obs['cell_iname']
    y_one = y_one.groupby('cell_iname').mean()
    Y = pd.concat([y_one, y_zero.loc[cell_lines_untreat]])

    W = pd.DataFrame([int(x in cell_lines_treat) for x in Y.index])
    W.index = Y.index

    treat_split = {}
    untreat_split = {}

    if split == 'train':
        task_splits = ['train', 'val']
        test_size = 0.1
    elif split == 'val':
        if include_support_in_val:
            task_splits = ['train', 'test']
            test_size = 0.5
        else:
            task_splits = ['test']
            test_size = 1.0
    elif split == 'test':
        task_splits = ['train', 'test']
        test_size = 0.5

    if test_size != 1.0:
        if len(cell_lines_treat) > 1:
            treat_split['train'], treat_split['test'] = train_test_split(cell_lines_treat, test_size=test_size, random_state=random_state)
        else:
            treat_split['train'] = cell_lines_treat
            treat_split['test'] = []
    else:
        treat_split['train'] = []
        treat_split['test'] = cell_lines_treat



    if test_size != 1.0:
        untreat_split['train'], untreat_split['test'] = train_test_split(cell_lines_untreat, test_size=test_size, random_state=random_state)
    else:
        untreat_split['train'] = []
        untreat_split['test'] = cell_lines_untreat

    if split == 'train':
        treat_split['val'] = treat_split.pop('test')
        untreat_split['val'] = untreat_split.pop('test')

    taskdata = {}
    taskdata = {}
    taskdata['train'] = {}
    taskdata['val'] = {}
    taskdata['test'] = {}

    for task_split in ['train', 'val', 'test']:

        # If the task split does not exist initialize to empty and move on
        if task_split not in task_splits:
            taskdata[task_split]['cell_id'] = []
            taskdata[task_split]['X'] = np.array(None)
            taskdata[task_split]['W'] = np.array(None)
            taskdata[task_split]['y'] = np.empty(None)
            taskdata[task_split]['tau'] = np.empty(None)
            continue

        if task_split == 'train':
            y0_flag = 'real'
            # y0_flag = 'impute'
        else:
            y0_flag = 'real'
        cell_lines_split = treat_split[task_split] + untreat_split[task_split]
        tau = get_tau(cell_lines_split, saved_y0_models, y_zero,
                      Y.loc[cell_lines_split], W.loc[cell_lines_split], feats,
                      y0_flag=y0_flag)
        taskdata[task_split]['cell_id'] = cell_lines_split
        taskdata[task_split]['X'] = feats.loc[cell_lines_split].values
        taskdata[task_split]['W'] = W.loc[cell_lines_split,0].values
        taskdata[task_split]['y'] = Y.loc[cell_lines_split].values
        taskdata[task_split]['tau'] = tau.loc[cell_lines_split].values

        if keep_treatment_col:
            if taskdata[task_split]['X'] is None:
                 continue
            T = np.where(all_tasks==task)[0][0]
            treat_cols = np.zeros([len(cell_lines_split), len(all_tasks)])
            treat_cols[:,T] = taskdata[task_split]['W']
            taskdata[task_split]['X'] = np.concatenate([taskdata[task_split]['X'],  treat_cols],1)
	 
    return taskdata

def main(args):
    """ Training script for deep models-based for CATE predition
        All arguments to the main() function are hyperparameters that can be set in the wandb sweep
    """

    test_str =""
    if args.TEST_MODE:
        test_str = "-test"
        performance_logs = {}

    run = wandb.init(config=args,
        project="lincs" + test_str,
        entity='dsp',
        name=args.experiment_name
    )

    logWANDBRun(run)

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


    if args.drug_embed_type == 'rdkit':
        ids_path = f"{args.folder_path}rdkit_2d_normalized_pertids.pkl"
        embeddings_path = f"{args.folder_path}rdkit_2d_normalized_embeddings_pertids.pkl"
    else:
        raise ValueError("Embedding type not implemented yet")




    # Separate drugs for train,val,test
    adata = sc.read_h5ad(
        '/dfs/project/perturb-gnn/L1000/level5_beta_trt_filtered_split_'+
         args.split_name+'.h5ad')
    CCLE_feats = pd.read_pickle(
        '/dfs/project/perturb-gnn/L1000/CCLE_feats.pkl')
    gene_mean = pd.read_csv(
             '/dfs/project/perturb-gnn/L1000/level5_beta_trt_filtered_split_'+
              args.split_name+'_mean_tau.csv', index_col=0)
    y_zero = pd.read_csv('/dfs/project/perturb-gnn/L1000/y_zeros.csv',
                         index_col=0)
    saved_y0_models = np.load('/dfs/project/perturb-gnn/L1000/y_zero_models'
                              '.npy', allow_pickle=True).item()

    train_adata = adata[adata.obs['split']=='Train']
    val_adata = adata[adata.obs['split'] == 'Val']
    test_adata = adata[adata.obs['split'] == 'Test']

    n20_most_diff_genes_index = get_DE_gene_idx(train_adata, 20)
    n50_most_diff_genes_index = get_DE_gene_idx(train_adata, 50)


    print("Train drugs/tasks: ", train_adata.obs['pert_id'].nunique())
    print("Validation drugs/tasks: ", val_adata.obs['pert_id'].nunique())
    print("Test drugs/tasks: ", test_adata.obs['pert_id'].nunique())

    print("Train cell lines: ", train_adata.obs['cell_iname'].nunique())
    print("Validation cell lines: ", val_adata.obs['cell_iname'].nunique())
    print("Test cell lines: ", test_adata.obs['cell_iname'].nunique())

    if args.wandb_online:
        wandb.log({"Number of train drugs": train_adata.obs['pert_id'].nunique(),
               "Number of val drugs": val_adata.obs['pert_id'].nunique(),
               "Number of test drugs": test_adata.obs['pert_id'].nunique()})

        wandb.log({"Number of train cell lines": train_adata.obs['cell_iname'].nunique(),
               "Number of val cell lines": val_adata.obs[
               'cell_iname'].nunique(),
               "Number of test cell lines": test_adata.obs[
               'cell_iname'].nunique()})

    # Device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device {device}')

    ####
    #### TODO initialize LINCsDataLoader
    ####
    cell_loader = CellDataLoader(DATA_LOCATION)

    if args.TEST_MODE:
        val_tasks = test_tasks
        splitStr = 'test'
    else:
        splitStr = 'val'


    te_mapper = TaskEmbeddingMapper(
        ids_path=ids_path,
        embeddings_path=embeddings_path,
        random=args.set_task_embeddings_to_random)

    task_embedding_config = args.task_embedding_config
    use_task_embeddings = args.use_task_embeddings

    input_dim = CCLE_feats.shape[1]
    task_dim = te_mapper(adata.obs['pert_id'].values[0]).shape[0]
    output_dim = adata.X.shape[1]

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
                    final_activation=args.use_final_activation,
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
                    residual = args.residual,
             #       final_activation=args.use_final_activation,

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
                    final_activation=args.use_final_activation,

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
                residual = args.residual,
                final_activation=args.use_final_activation,

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

    train_drugs = list(set(train_adata.obs['pert_id'].values))
    val_drugs = list(set(val_adata.obs['pert_id'].values))
    test_drugs = list(set(test_adata.obs['pert_id'].values))

    np.random.shuffle(train_drugs)
    train_cell_lines = train_adata.obs['cell_iname'].unique()

    tasks_observed = []


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
    
            # Gather losses over meta batch in a list and later average
            meta_train_loss = 0.0

            meta_train_losses_start = []
            meta_train_losses_end = []
            meta_train_losses_avg = []
    
            meta_train_losses_val = []
            meta_train_losses_val_base_easy = []
            meta_train_losses_val_base_hard = []
    
            meta_train_losses_val_diff20 = []
            meta_train_losses_val_diff20_base_easy = []
            meta_train_losses_val_diff20_base_hard = []
    
            meta_train_losses_val_diff50 = []
            meta_train_losses_val_diff50_base_easy = []
            meta_train_losses_val_diff50_base_hard = []


            tasks_train = []

            for i in range(0,1):

                TASK_DRUG = np.random.choice(train_drugs,1)[0]
                print("Sampled task with drug: {}".format(TASK_DRUG))
                taskdata = LINCSDataLoader(task=TASK_DRUG,split='train',keep_treatment_col=args.s_learner,include_support_in_val=(args.s_learner or args.all_tasks_baseline), adata=adata,saved_y0_models=saved_y0_models, y_zero=y_zero, feats=CCLE_feats)
                subsets = taskdata#['subsets']

                subsets['train']['X'] = subsets['train']['X'].astype(np.float32)
                subsets['val']['X'] = subsets['val']['X'].astype(np.float32)

                X_train = subsets['train']['X']
                X_val = subsets['val']['X']

                

                tau_train = subsets['train']['tau']
                tau_val = subsets['val']['tau']
                train_treatment = subsets['train']['W']
                val_treatment = subsets['val']['W']
                test_treatment = subsets['test']['W']
        
                ### ONLY THE TREATMENT PATIENTS
                input_train = X_train[ subsets['train']['W'].nonzero()[0],:]
                input_val = X_val[subsets['val']['W'].nonzero()[0],:]
        
                target_train = tau_train[ subsets['train']['W'].nonzero()[0]].astype(np.float32)
                target_val = tau_val[subsets['val']['W'].nonzero()[0]].astype(np.float32)

                task_embedding = te_mapper(TASK_DRUG)

                numRowsEmbedding = input_train.shape[0]
                numRowsEmbeddingVal = input_val.shape[0]
                treatmentEmbeddingDim = task_embedding.shape[0]

                treatment_matrix = np.broadcast_to(task_embedding,(numRowsEmbedding,treatmentEmbeddingDim))
                treatment_matrix_val = np.broadcast_to(task_embedding,(numRowsEmbeddingVal,treatmentEmbeddingDim))

                tasks_observed.append(TASK_DRUG)
                print(TASK_DRUG)
                tasks_train.append((TASK_DRUG,input_train, input_val, target_train, target_val, treatment_matrix, treatment_matrix_val))



            #breakpoint()
                                                # input                               #embedding                              # target
            train_dataset = BioSINTaskDataset(np.concatenate([x[1] for x in tasks_train]), np.concatenate([x[5] for x in tasks_train]), np.concatenate([x[3] for x in tasks_train]))
            val_dataset = BioSINTaskDataset(np.concatenate([x[2] for x in tasks_train]), np.concatenate([x[6] for x in tasks_train]), np.concatenate([x[4] for x in tasks_train]))

            sampler = RandomSampler(train_dataset,replacement=True,num_samples=args.caml_k*args.batch_size)
            train_loader = DataLoader(
                    train_dataset, 
                    batch_size = args.batch_size, 
                    num_workers=0,
                    shuffle=True
            )

            val_loader = DataLoader(val_dataset, batch_size = args.batch_size, num_workers=0)
        
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
                flatten=False
            )
        
            # Get new optimizer state:
            inner_optimizer_state = inner_optimizer.state_dict()
            # Update gradient of outer model for SGD: Add Phi tilde component
            for p, l in zip(model.parameters(), learner.parameters()):
                p.grad.data.add_(-1.0, l.data )
            meta_train_loss += inner_val_loss.item()

    
            logs = {"iteration": iteration, "tasks_observed":len(tasks_observed), "tasks_observed_unique":len(set(tasks_observed)), "task": 0, "inner_val_loss": inner_val_loss, "task_drug": TASK_DRUG}
            

            wandb.log(logs)
    
    
            del taskdata
            del train_loader
            if args.sin_learner == False and args.graphite_learner == False:
                del val_loader
                del val_dataset
            del train_dataset
                
    
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

            mse_diff20 = 0
            mse_diff50 = 0
            mse_denom = 0

            tau_reals = []
            tau_preds_all = []

            counter = 0 
            for val_task in val_drugs:
                print("{}/{}".format(counter,len(val_drugs)))
                counter+=1
                ###
                ### Migrate the validation code 
                ###

                valData = LINCSDataLoader(task=val_task, split=splitStr,keep_treatment_col=((args.s_learner or args.single_task_baseline) and args.r_learner==False),include_support_in_val=(args.s_learner or args.single_task_baseline or args.all_tasks_baseline), adata=adata,saved_y0_models=saved_y0_models, y_zero=y_zero, feats=CCLE_feats)
                subsets = valData
                X_val = subsets['test']['X'].astype(np.float32)
                y_val = subsets['test']['y'].astype(np.float32)
                val_treatment = subsets['test']['W']

                if args.TEST_MODE:
                    iter_log = {}
                    print("Testing for task {}".format(valData['task']))

                    num_treatments = float(subsets['test']['W'].sum())
            
                    if num_treatments < 1:
                        print("Skipping task {} because only 0 treated cells".format(test_task))
                        continue
            
                    num_treatments_all.append(num_treatments)
                    num_positive_all.append(0)
                    test_tasks_all.append(val_task)

                if args.sin_learner == False and args.graphite_learner == False and args.s_learner_h == False:
                    if args.single_task_baseline == False:
    
                        if use_task_embeddings:
                            # Load the drug embedding for the task
                            task_embedding = te_mapper(val_task)
                            task_embedding = task_embedding[None, :]
        
                            do_concat = task_embedding_config == 'early_concat'
                            meta_val_dataset = BioTaskDataset(X_val, task_embedding, do_concat, y_val)
                        else:
        
                            if args.s_learner == False:
        
                                meta_val_dataset = BioDataset(X_val, y_val)
                            else:

                                ###
                                ### BIOTODO: Correct S-learner implemenation
                                ###
                                trt_columns = [vocab_json['Drugs'].index(x) for x in val_task.split("_")]
                                for trt_column in trt_columns:
                                    X_val[:,trt_column] = 1
        
                                X_val_0 = X_val.copy()
        
                                for trt_column in trt_columns:
                                    X_val_0[:,trt_column] = 0
        
                                meta_val_dataset = BioSLearnerDataset(X_val, X_val_0, y_val)
        
                        if args.s_learner == False:
                            meta_val_loader = DataLoader(meta_val_dataset, batch_size = args.batch_size, num_workers=0)
                        else:
                            meta_val_loader = DataLoader(meta_val_dataset, batch_size = args.batch_size, num_workers=0)
        
        
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
        
                                tau_preds.append(tau_pred)
        
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
                            meta_tau_pred = torch.tensor(meta_learner.predict(X_val))
    
                        else:
    

                            from catenets.models.torch.representation_nets import TARNet, DragonNet
                            from catenets.models.torch.flextenet import FlexTENet
                            import catenets.logger as log
                            import catenets.models.constants

                            log.add(sys.stdout,level=0)
  
                            if args.dl_learner_type == 'TARNet':
                                meta_learner = TARNet(n_unit_in=input_dim,n_iter_print=1,batch_size=args.batch_size,n_iter=args.dl_n_iter, n_layers_r=args.dl_n_layers_r,n_units_r=args.dl_n_units_r,n_layers_out=args.dl_n_layers_out,n_units_out=args.dl_n_units_out,weight_decay=args.dl_weight_decay,lr=args.dl_lr,dropout=args.dl_dropout , penalty_disc=args.dl_penalty_disc)
    
                            if args.dl_learner_type == 'DragonNet':
                                meta_learner = DragonNet(n_unit_in=input_dim,n_iter_print=1,batch_size=args.batch_size,n_iter=args.dl_n_iter, n_layers_r=args.dl_n_layers_r,n_units_r=args.dl_n_units_r,n_layers_out=args.dl_n_layers_out,n_units_out=args.dl_n_units_out,weight_decay=args.dl_weight_decay,lr=args.dl_lr,dropout=args.dl_dropout  )
            
                            if args.dl_learner_type == 'FlexTENet':
                                meta_learner = FlexTENet(binary_y=True,n_unit_in=input_dim,n_iter_print=1,batch_size=args.batch_size,n_iter=args.dl_n_iter, n_layers_r=args.dl_n_layers_r,n_layers_out=args.dl_n_layers_out,weight_decay=args.dl_weight_decay,lr=args.dl_lr,dropout=args.dl_dropout ,   n_units_s_out=args.dl_n_units_s_out, n_units_p_out=args.dl_n_units_p_out, n_units_s_r=args.dl_n_units_s_r,n_units_p_r=args.dl_n_units_p_r, private_out=args.dl_private_out, penalty_orthogonal=args.dl_penalty_orthogonal)
        
                            meta_learner.fit(subsets['train']['X'], subsets['train']['y'], subsets['train']['W'])
                            
                            numRowsEmbedding = X_val.shape[0]

                            tau_preds = []
                            for i in range(0,20):
                                start_i = i*50000
                                print(start_i)
                                end_i = min(numRowsEmbedding, (i+1)*50000)


                                tau_pred = torch.tensor(meta_learner.predict(X_val[start_i:end_i,:])).to(device)
                                tau_preds.append(tau_pred)

                                if end_i == numRowsEmbedding:
                                    break

                            meta_tau_pred = torch.cat(tau_preds)


                elif args.sin_learner == True or args.graphite_learner == True or args.s_learner_h == True:
                    task_embedding = te_mapper(val_task)
                    null_task_embedding = te_mapper('NULL')
    
                    numRowsEmbedding = X_val.shape[0]
                    treatmentEmbeddingDim = task_embedding.shape[0]
    
                    treatment_matrix = np.broadcast_to(val_treatment, (treatmentEmbeddingDim,numRowsEmbedding)).T
                    y1_t_matrix = np.broadcast_to(task_embedding,(numRowsEmbedding,treatmentEmbeddingDim))
                    y0_t_matrix = np.broadcast_to(null_task_embedding,(numRowsEmbedding,treatmentEmbeddingDim))
    
                    treatment_features_matriX_val = np.multiply(treatment_matrix,y1_t_matrix) + np.multiply(1-treatment_matrix,y0_t_matrix)

                    covariates_test, treatment_node_features_test, target_outcome_test = X_val,torch.Tensor(treatment_features_matriX_val),y_val

                    tau_preds = []

                    for i in range(0,20):
                        start_i = i*50000
                        print(start_i)
                        end_i = min(numRowsEmbedding, (i+1)*50000)


                        covariates_test_subset  = torch.Tensor(covariates_test[start_i:end_i,:]).to(device)
                        treatment_node_features_test_subset  = treatment_node_features_test[start_i:end_i,:].to(device)
                        target_outcome_test_subset  = torch.Tensor(target_outcome_test[start_i:end_i]).to(device)
    
    
                        treatment_node_features_test_y1  = torch.Tensor(y1_t_matrix[start_i:end_i,:]).to(device)
                        treatment_node_features_test_y0  = torch.Tensor(y0_t_matrix[start_i:end_i,:]).to(device)
    
    
                        if args.sin_learner == True:
                            allTestData = (covariates_test_subset, treatment_node_features_test_subset, target_outcome_test_subset)
        
                            with torch.no_grad():
                                tau_pred = torch.tensor(model.test_prediction(allTestData))
                                tau_preds.append(tau_pred)
    
                        elif args.graphite_learner == True : 
                            allTestData_y1 = (covariates_test_subset, treatment_node_features_test_y1, target_outcome_test_subset)
                            allTestData_y0 = (covariates_test_subset, treatment_node_features_test_y0, target_outcome_test_subset)
    
                            with torch.no_grad():
                                tau_pred = torch.tensor(model.test_prediction(allTestData_y1)) -  torch.tensor(model.test_prediction(allTestData_y0)) 
                                tau_preds.append(tau_pred)
    
                        elif args.s_learner_h == True:
    
                            with torch.no_grad():

                                tau_pred = torch.tensor(model(covariates_test_subset, treatment_node_features_test_y1)) -  torch.tensor(model(covariates_test_subset, treatment_node_features_test_y0)) 
                                tau_preds.append(tau_pred)

                        if end_i == numRowsEmbedding:
                            break

                    meta_tau_pred = torch.cat(tau_preds)


                 ###
                 ### BIOTODO: Correct evaluation metrics
                 ###


                if subsets['test']['W'].sum() == 0:
                    print("SKIPPING BECAUSE NO TREATMENT")
                    continue

                #breakpoint()


                tau_reals.append(subsets['test']['tau'][subsets['test']['W'].nonzero()[0]])
                tau_preds_all.append(meta_tau_pred.cpu().numpy()[subsets['test']['W'].nonzero()[0]])
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


            tau_real = np.concatenate(tau_reals)
            tau_pred = np.concatenate(tau_preds_all)
            tau_pred_easy = np.zeros_like(tau_pred)
            tau_pred_hard = torch.tensor(gene_mean.T.values).repeat( tau_pred.shape[0], 1).numpy()

            iter_log['val_mse_diff_20'] = mean_squared_error(tau_real[:,n20_most_diff_genes_index], tau_pred[:,n20_most_diff_genes_index]) 
            iter_log['val_mse_diff_50'] = mean_squared_error(tau_real[:,n50_most_diff_genes_index], tau_pred[:,n50_most_diff_genes_index]) 

            iter_log['val_mse_diff_20_hard'] = mean_squared_error(tau_real[:,n20_most_diff_genes_index], tau_pred_hard[:,n20_most_diff_genes_index]) 
            iter_log['val_mse_diff_50_hard'] = mean_squared_error(tau_real[:,n50_most_diff_genes_index], tau_pred_hard[:,n50_most_diff_genes_index]) 

            iter_log['val_mse_diff_20_easy'] = mean_squared_error(tau_real[:,n20_most_diff_genes_index], tau_pred_easy[:,n20_most_diff_genes_index]) 
            iter_log['val_mse_diff_50_easy'] = mean_squared_error(tau_real[:,n50_most_diff_genes_index], tau_pred_easy[:,n50_most_diff_genes_index]) 

            result_dict = {}

            if args.TEST_MODE == False:

                for metricName, metricValue in iter_log.items():

                    if 'easy' in metricName or 'hard' in metricName:
                        continue
    
                    print(metricName, metricValue)
 
                    bv_log(
                        name = metricName,
                        value = metricValue,
                        step = iteration,
                        increasing = False
                    )

                # log best values
                wandb.log(
                    bv_log.best_values
                )

                torch.save(model, os.path.join(getWANDBDir(run), "model_{}.pt".format(iteration)))


        if args.TEST_MODE == False:
            wandb.log(
                iter_log
            )

        if args.single_task_baseline or args.TEST_MODE:
            break


def print_memory():
    x = torch.cuda.memory_allocated()
    print(f'CUDA MEMORY USAGE : {x/(10**9)} GB')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--folder_path', type=str,
                        default="/dfs/scratch0/asurina/ehr-drug-sides/BIO_preprocessing/BIO_datasets/", help="Folder where data is stored") 
    parser.add_argument('--tasks', type=str,
                        default="tasks/", help="Tasks folder name")  # "tasks_pertids_full_unnorm/"

    parser.add_argument('--experiment_name', type=str, default='cluster_split')
    parser.add_argument('--wandb_online', type=str,
                        default="true", help="Sync wandb online")


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

    # Task embeddings config
    parser.add_argument('--set_task_embeddings_to_random',
                        type=str, default="false")
    parser.add_argument('--task_embedding_config', type=str, default="late_concat_layernorm",
                        choices=['early_concat', 'late_concat',
                                 'late_concat_layernorm'],
                        help='How to include task embeddings')
    parser.add_argument('--drug_embed_type', type=str, default='rdkit',
                        choices=['KG', 'Morgan', 'rdkit', "ESPF",
                                 "ErG", "Daylight", "MACCSKeysFingerprint"],
                        help='Which embeddings for tasks (drugs) to use') #For current dataset only rdkit is available



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
    parser.add_argument('--use_final_activation', type=str,  default="false")  # if tau is in range [-1,1]

    parser.add_argument('--independence_regularisation_coeff', type=float, default=0.01) 
    parser.add_argument('--dim_output_treatment', type=int, default=256) 
    parser.add_argument('--dim_output_covariates', type=int, default=256) 
    parser.add_argument('--num_final_ff_layer', type=int, default=4) 

    parser.add_argument('--kill_seconds', type=int, default=172800) 
    parser.add_argument('--split_name', type=str, default='cluster')


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
