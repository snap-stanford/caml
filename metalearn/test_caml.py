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
from causal_singletask.models.xlearner import (XTrtLearner, TLearner,XLearner,RLearner)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from IPython import embed
from pytorch_lightning.loggers import WandbLogger
import sys 
from typing import Union
from scipy.sparse import csr_matrix, coo_matrix

# Custom modules:
from causal_singletask.consts_and_utils import *
import causal_singletask.models
from causal_singletask import models
from causal_singletask.torch_utils import BestValueLogger, EarlyStopping
from metalearn.eval import treatmentRecall, addLogDict, treatmentRATE, countPositiveLabels
from metalearn.parse import PatientDataLoader, XLearnerM0, QueryPatientsByDrug
from metalearn.caml import CAMLAdapter
from metalearn.task_utils import (TaskEmbeddingMapper, ClaimsTaskDataset, 
    task_embedding_paths, sparse_batch_collate, filter_for_used_drugs)
from pdb import set_trace as bp
from metalearn.eval import treatmentRecall, treatmentPrecision, addLogDict, treatmentRATE, treatmentChoose2
from metalearn.parse import PatientDataLoader, XLearnerM0
from metalearn.caml import CAMLAdapter
from metalearn.task_utils import DrugEmbeddingMapper, sparse_batch_collate, sparse_batch_collate2

import catenets.logger as log
from catenets.models.torch.representation_nets import TARNet, DragonNet
from catenets.models.torch.flextenet import FlexTENet


def main(args):
    """ Training script for deep models-based for CATE predition 
        All arguments to the main() function are hyperparameters that can be set in the wandb sweep
    """


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device {device}')

    patientloader = PatientDataLoader(DATA_LOCATION)

    df_val_test_split = pd.read_csv(args.val_test_location)
    test_tasks = list(df_val_test_split[df_val_test_split['split']=='test']['DrugBankID'])

    vocab_json = json.loads(open(VOCAB_JSON,'r').read())

    if args.include_combos:
        df_tasks = pd.read_csv(TASKS_CSV)
        df_tasks = df_tasks[(df_tasks['drug1'].isin(test_tasks)) & (df_tasks['drug2'].isin(test_tasks))]
        test_tasks = test_tasks + list(df_tasks['Drugs'])

    print('testing these tasks: {}'.format(str(test_tasks)))
    
    performance_logs = {}

    num_treatments_all = []
    num_positive_all = []
    test_tasks_all = []

    for test_task in test_tasks:

        testData = patientloader(test_task, split='test')
        subsets = testData['subsets']
        X_test = subsets['test']['X'].astype(np.float32)
        y_test = torch.tensor(subsets['test']['y'].astype(np.float32))
        test_treatment = subsets['test']['W']

        num_treatments = float(subsets['test']['W'].sum())
        num_positive = float(y_test[subsets['test']['W'].nonzero()[0]].sum())

        if num_positive < 5:
            print("Skipping task {} because only {} positive samples".format(test_task, num_positive))
            continue

        num_treatments_all.append(num_treatments)
        num_positive_all.append(num_positive)
        test_tasks_all.append(test_task)


        #####
        ##### Generate predictions for zero shot model 
        #####
    
        models_to_test = []

        if args.caml:
            models_to_test.append('caml')

        if args.x_trt_learner:
            models_to_test.append('xtrt')

        if args.s_learner:
            models_to_test.append('s_learner')

        if args.t_learner:
            models_to_test.append('t_learner')

        if args.x_learner:
            models_to_test.append('x_learner')

        if args.r_learner:
            models_to_test.append('r_learner')

        if args.TARNet:
            models_to_test.append('TARNet')

        if args.FlexTENet:
            models_to_test.append('FlexTENet')

        if args.DragonNet:
            models_to_test.append('DragonNet')


        for modelName in models_to_test:

            print("EVALUATING TASK {} WITH MODEL {}".format(test_task,modelName))
    
            performance_log = {}
    
            if modelName == 'caml' or modelName == 's_learner':

                print("Testing model at location {}".format(args.model_location))
            
                # Init mapping from drug to task embedding:
                te_mapper = TaskEmbeddingMapper(
                    embedding_path = task_embedding_paths[args.kge], 
                    aggregation = args.task_embedding_aggregation
                )
                print("Loading model at location {}".format(args.model_location))
                # Model class must be defined somewhere
                model = torch.load(args.model_location, map_location=torch.device('cpu'))
                model.to(device)
                model.eval()
            
                total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f'Loaded Model with {total_params} parameters.')
                print(model)
            
                if modelName == 'caml':
                    if args.use_task_embeddings:
                        task_embedding = te_mapper(test_task) 
                        task_embedding = task_embedding[None, :]
                        print(f' Using task embedding configuration: {args.task_embedding_config}')
                        do_concat = args.task_embedding_config == 'early_concat'
                        meta_test_dataset = ClaimsTaskDataset(X_test, task_embedding, do_concat, y_test) 
                    else:
                        print(f'Not using task embeddings.')
                        meta_test_dataset = ClaimsDataset(X_test, y_test)
            
                    meta_test_loader = DataLoader(meta_test_dataset, batch_size = args.batch_size, num_workers=0,  collate_fn=sparse_batch_collate)

                elif modelName == 's_learner':
                        trt_columns = [vocab_json['Drugs'].index(x) for x in test_task.split("_")]
                        for trt_column in trt_columns:
                            X_test[:,trt_column] = 1

                        X_test_0 = X_test.copy()

                        for trt_column in trt_columns:
                            X_test_0[:,trt_column] = 0

                        meta_test_dataset = ClaimsSLearnerDataset(X_test, X_test_0, y_test)
                        meta_test_loader = DataLoader(meta_test_dataset, batch_size = args.batch_size, num_workers=0,  collate_fn=sparse_batch_collate2)



                with torch.no_grad():
                
                    model.eval()
                    n_samples = 0
                    tau_preds = []
                
                    for eval_iter, batch in enumerate(tqdm(meta_test_loader)):

                        if modelName == 'caml':
                            if args.task_embedding_config in ['late_concat','late_concat_layernorm']:
                                data, task_embedding, labels = batch  
                                data  = data.to(device) 
                                task_embedding = task_embedding.to(device)
                                tau_pred = model(data, task_embedding)
                                task_embedding.detach()
                            else:
                                data, labels = batch  
                                data = data.to(device)
                                tau_pred = model(data)

                        elif modelName == 's_learner':
                            data,data2, labels = batch
                            data = data.to(device)
                            data2 = data2.to(device)       
                            tau_pred = model(data) - model(data2)                         

                        tau_preds.append(tau_pred.view(-1))
                
                    meta_tau_pred = torch.cat(tau_preds).cpu().numpy()
        
            elif modelName == 'xtrt':
                print("Evaluating baseline")
    
                xtrt = XTrtLearner(args=args)
                xtrt.train(subsets['train'])
                meta_tau_pred = xtrt.predict(X_test)

            elif modelName == 't_learner':
                print("Evaluating baseline")
    
                meta_learner = TLearner(args=args)
                meta_learner.train(subsets['train'])
                meta_tau_pred = torch.tensor(meta_learner.predict(X_test))

            elif modelName == 'x_learner':
                print("Evaluating baseline")
    
                meta_learner = XLearner(args=args)
                meta_learner.train(subsets['train'])
                meta_tau_pred = torch.tensor(meta_learner.predict(X_test))

            elif modelName == 'r_learner':
                print("Evaluating baseline")
    
                meta_learner = RLearner(args=args)
                meta_learner.train(subsets['train'])
                meta_tau_pred = torch.tensor(meta_learner.predict(X_test))

            elif modelName == 'FlexTENet':
                print("Evaluating baseline")

                meta_learner = FlexTENet(binary_y=True,n_unit_in=input_dim,n_iter_print=1,batch_size=args.batch_size,n_iter=args.dl_n_iter)
                meta_learner.fit(subsets['train']['X'], subsets['train']['y'], subsets['train']['W'])
                meta_tau_pred = torch.tensor(meta_learner.predict(X_test))

            elif modelName == 'DragonNet':
                print("Evaluating baseline")

                meta_learner = DragonNet(n_unit_in=input_dim,n_iter_print=1,batch_size=args.batch_size,n_iter=args.dl_n_iter)
                meta_learner.fit(subsets['train']['X'], subsets['train']['y'], subsets['train']['W'])
                meta_tau_pred = torch.tensor(meta_learner.predict(X_test))

            elif modelName == 'TARNet':
                print("Evaluating baseline")

                meta_learner = TARNet(n_unit_in=input_dim,n_iter_print=1,batch_size=args.batch_size,n_iter=args.dl_n_iter)
                meta_learner.fit(subsets['train']['X'], subsets['train']['y'], subsets['train']['W'])
                meta_tau_pred = torch.tensor(meta_learner.predict(X_test))


            recall_thresholds = [0.001,0.002,0.005,0.01]
            rate_thresholds = [0.001,0.002,0.005,0.01]
        
            # Compute eval metrics (recall, rate)
            table_precision = treatmentPrecision(
                    subsets['test'], 
                    list(meta_tau_pred),
                    recall_thresholds 
            )
            table_recall = treatmentRecall(
                    subsets['test'], 
                    list(meta_tau_pred),
                    recall_thresholds 
            )
            table_rate = treatmentRATE(
                    subsets['test'], 
                    list(meta_tau_pred), 
                    RandomForestClassifier(
                        random_state=0, 
                        n_estimators=150, 
                        max_depth=30,
                        n_jobs=75,
                        verbose=True), 
                    rate_thresholds 
            )
            table_choose2 = treatmentChoose2(subsets['test'], meta_tau_pred )
            
            # the first time, we collect the metric names for best value logger
            addLogDict(performance_log, table_precision,test_task,split='Test',metric_name='precision')
            addLogDict(performance_log, table_recall,test_task,split='Test',metric_name='recall')
            addLogDict(performance_log, table_rate,test_task,split='Test',metric_name='rate')
            addLogDict(performance_log, table_choose2, test_task, split='Test',metric_name='choose2')
            
            performance_logs[(modelName,test_task)] = performance_log

        
            performance_log_loc = MODEL_OUTPUTS+ 'performance_logs/'
            os.makedirs(performance_log_loc,exist_ok=True)
            performance_logs_final = logsToTable(performance_logs).merge(pd.DataFrame({'task':  test_tasks_all,'numPositive':num_positive_all, 'numTreatments':num_treatments_all}),on='task')
            performance_logs_final.to_csv(performance_log_loc+str("_".join(models_to_test))+"__"+args.val_test_location.split('/')[-1]+"__"+args.output_suffix)
            print(performance_logs_final)


 
def print_memory():
    x = torch.cuda.memory_allocated()
    print(f'CUDA MEMORY USAGE : {x/(10**9)} GB')

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='DB00563-Pancytopenia')
    # Training config:
    parser.add_argument('--batch_size', type=int,
                        default=2048*4)
    parser.add_argument('--val_test_location', type=str,  default='metalearn/splits/new_splits.csv')
    parser.add_argument('--task_dir', type=str, default='task_embedding/tasks/', help='reative path to task infos (embeddings, drugs) from repo root')
    parser.add_argument('--use_task_embeddings', type=str, default="true") # hack to cope with wandb
    parser.add_argument('--task_embedding_config', type=str, default="early_concat",
                        choices=['early_concat', 'late_concat','late_concat_layernorm'], help='How to include task embeddings')
    parser.add_argument('--model_location')
    parser.add_argument('--kge', type=str, default="stargraph", 
                        help='What KG embedding to use [distmult, stargraph]')
    parser.add_argument('--task_embedding_aggregation', type=str, default="sum", 
                        help='Aggregation function for task embeddings (multiple drugs) [sum, mean]')

    parser.add_argument('--caml', type=str, default="false", help='Should we add main camel model')
    parser.add_argument('--x_trt_learner', type=str, default="false", help='Should we add X-trt baseline')
    parser.add_argument('--s_learner', type=str, default="false", help='Should we add S-learner baseline') 
    parser.add_argument('--t_learner', type=str, default="false", help='Should we add T-learner baseline') 
    parser.add_argument('--x_learner', type=str, default="false", help='Should we add X-learner baseline') 
    parser.add_argument('--r_learner', type=str, default="false", help='Should we add R-learner baseline') 

    parser.add_argument('--TARNet', type=str, default="false", help='Should we add TARNet baseline') 
    parser.add_argument('--FlexTENet', type=str, default="false", help='Should we add FlexTENet baseline') 
    parser.add_argument('--DragonNet', type=str, default="false", help='Should we add DragonNet baseline') 

    parser.add_argument('--include_combos', type=str, default="false", help='Should we test all combinations of the drugs')
    parser.add_argument('--output_suffix', type=str, default="", help='Concatenate this to the end of the file')
    parser.add_argument('--rf_n_estimators', type=int, default=150) 
    parser.add_argument('--rf_max_depth', type=int, default=30) 
    parser.add_argument('--rf_min_samples_split', type=int, default=2) 
    parser.add_argument('--rf_criterion_regress', type=str, default="squared_error") 
    parser.add_argument('--rf_criterion_binary', type=str, default="gini") 
    parser.add_argument('--rf_max_features', type=str, default="sqrt") 
    
    parser.add_argument('--dl_n_iter', type=int, default=100) 

    args = parser.parse_args()
    # update pseudo-bool arguments:
    args_to_convert = [
            key for key,val in vars(args).items() if val in ["true", "false"]
    ]  #['batch_norm', 'use_lr_scheduler', 'use_task_embeddings', 'residual', 'task_sample_weighting']
    args = convert_str_to_bool(args, args_to_convert)
    print(f'Boolean args: {args_to_convert}')
    main(args) 



#print('Reached end of loop---- {}------'.format(task))
#print(torch.cuda.memory_allocated())
#print(torch.cuda.max_memory_allocated())
#print('Reached end of loop---- {}  --')

