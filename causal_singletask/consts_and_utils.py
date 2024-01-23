import os
import gzip
import json
import scipy
import random
import glob
import math
import numpy as np
import pickle as pkl
import pandas as pd
from IPython.core.debugger import Pdb; 

import pathlib
import hashlib
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
#from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_predict, KFold, train_test_split
from scipy.sparse import hstack
import pickle
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import os.path
from IPython.core.debugger import Pdb; 
from torch.utils.data import Dataset 
from typing import Union
from scipy.sparse import coo_matrix, csr_matrix
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from IPython.core.debugger import Pdb;
from IPython import embed
from pdb import set_trace as bp


MODEL_OUTPUTS = './model_outputs/'
DATA_LOCATION = '/home/anon/ehr-drug-sides/data/'
VOCABULARY = '/home/anon/ClaimsCausal/BigQuery/vocab.pickle'
TASKS_CSV = '/home/anon/ehr-drug-sides/task_embedding/tasks/drug_tasks_new.csv'
VOCAB_JSON = '/home/anon/ehr-drug-sides/metalearn/BigQuery/vocab.json'
DATABASE_LOCATION = '/home/anon/ehr-drug-sides/data/db'

class ClaimsDataset(Dataset):
    """
    Custom Dataset class for scipy sparse matrix
    """
    def __init__(self, data:Union[np.ndarray, coo_matrix, csr_matrix], 
                 *tensors: torch.Tensor, 
                 ):
        
        assert all(data.shape[0] == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
        
        # Transform data coo_matrix to csr_matrix for indexing
        if type(data) == coo_matrix:
            self.data = data.tocsr()
        else:
            self.data = data
            
        self.tensors = tensors
        

    def __getitem__(self, index:int):
        return (self.data[index], *[tensor[index] for tensor in self.tensors])

    def __len__(self):
        return self.data.shape[0]



class ClaimsSLearnerDataset(Dataset):
    """
    Custom Dataset class for scipy sparse matrix
    """
    def __init__(self, data:Union[np.ndarray, coo_matrix, csr_matrix], 
                       data2:Union[np.ndarray, coo_matrix, csr_matrix], 
                 *tensors: torch.Tensor, 
                 ):
        
        assert all(data.shape[0] == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
        
        # Transform data coo_matrix to csr_matrix for indexing
        if type(data) == coo_matrix:
            self.data = data.tocsr()
        else:
            self.data = data
            

        if type(data2) == coo_matrix:
            self.data2 = data2.tocsr()
        else:
            self.data2 = data2

        self.tensors = tensors
        

    def __getitem__(self, index:int):
        return (self.data[index], self.data2[index] , *[tensor[index] for tensor in self.tensors])

    def __len__(self):
        return self.data.shape[0]


def flatten(t):
    return [item for sublist in t for item in sublist]


def logsToTable(logs):
    log_list = []
    
    metricNames = []
    taskNames = []
    modelNames = []
    values = []
    thresholds = []
    
    for logKey, log in logs.items():

        modelName, taskName = logKey

        for metricName, value in log.items():
            metricNames.append(metricName.split(' ')[1])
            thresholds.append(metricName.split(' ')[3])
            taskNames.append(metricName.split(' ')[4].lstrip('(').rstrip(')'))
            modelNames.append(modelName)
            values.append(value)
    
    return pd.DataFrame({'metric':metricNames,'task':taskNames,'values':values, 'threshold':thresholds, 'model':modelNames})
            


def get_optimizer(model, lr, weight_decay):
        """Get optimizers."""
        # we don't apply weight decay to bias and layernorm parameters, as inspired by:
        # https://colab.research.google.com/github/PytorchLightning/pytorch-lightning/blob/master/notebooks/04-transformers-text-classification.ipynb
        no_decay = ["bias", "LayerNorm.weight"] # in case we upgrade to Transformer arch
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ] 
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=lr, weight_decay=weight_decay)
        return optimizer

def convert_str_to_bool(args_obj, arguments=['batch_norm', 'use_lr_scheduler']):
    for arg in arguments:
        # check format
        curr_value = getattr(args_obj, arg)
        try:
            assert curr_value in ['true', 'false']
        except:
            raiseValueError(f' Argument {arg}={curr_value} needs to provided as true or false')
        # set bool format
        bool_value = True if curr_value == 'true' else False
        setattr(args_obj, arg, bool_value)
    return args_obj

def predict(model, loader, device):
    full_Y0_hat = []

    with torch.no_grad():                
        for ix, (samples, labels) in enumerate(tqdm(loader)):            
            samples = samples.to(device)
            labels = labels.to(device)
            Y0_hat = model(samples).view(-1)
            full_Y0_hat.append(Y0_hat)
    return torch.cat(full_Y0_hat)

def flattenShards(task,data_path=DATA_LOCATION):
    files = list(glob.glob(data_path +"/tasks/{}/shards/*".format(task)))
    files.sort()
    
    print('loading all shards for task {}'.format(task))
    SHARDS_FINISH_PROCESSING = []
    for i_shard in files:
        print(i_shard)
        loaded_shard = None
        with open(i_shard, 'rb') as handle:
            loaded_shard = pickle.load(handle)
        SHARDS_FINISH_PROCESSING.append(loaded_shard)  

    if len(SHARDS_FINISH_PROCESSING) < 10: # modified from 70
        print('looking in {}'.format(data_path +"{}/shards/*".format(task)))
        raise ValueError('Only found {} shards in {}, supposed to find > 70'.format(len(SHARDS_FINISH_PROCESSING),task))

    if len(SHARDS_FINISH_PROCESSING) != 80:
        print('Only found {} shards in {}, supposed to find 80'.format(len(SHARDS_FINISH_PROCESSING),task))

    X = []
    for i in range(0,len(SHARDS_FINISH_PROCESSING[0]['X'])):
        X.append(scipy.sparse.vstack(x['X'][i] for x in SHARDS_FINISH_PROCESSING))
        
    Y = []
    for i in range(0,len(SHARDS_FINISH_PROCESSING[0]['Y'])):
        Y.append(scipy.sparse.vstack(x['Y'][i] for x in SHARDS_FINISH_PROCESSING))

    X = scipy.sparse.csr.csr_matrix(scipy.sparse.hstack(X))
    Y = scipy.sparse.csr.csr_matrix(scipy.sparse.hstack(Y))

    PATIDS = np.array(flatten(x['PATID'] for x in SHARDS_FINISH_PROCESSING))
    DS = np.array(flatten(x['DS'] for x in SHARDS_FINISH_PROCESSING))

    output = {
        'PATIDS': PATIDS,
        'DATESTAMPS': DS, 
        'X': X, 
        'Y': Y,
        'task':task
    }
    
    return output

def save_pickle(data, path):
    parent_dir = os.path.dirname(path)
    if not os.path.exists(parent_dir):
        os.system(f'mkdir -p {parent_dir}')
    pkl.dump(data, open(path, "wb"))

def split_data(data, ratio, seed):
    PATIDS_vocab = sorted(set(data['PATIDS']))
    random.seed(seed)
    random.shuffle(PATIDS_vocab)
    nums  = [int(len(PATIDS_vocab) * r) for r in ratio]
    PATIDS_train = set(PATIDS_vocab[:nums[0]])
    PATIDS_val   = set(PATIDS_vocab[nums[0]:nums[1]])
    PATIDS_test  = set(PATIDS_vocab[nums[1]:])
    return PATIDS_train, PATIDS_val, PATIDS_test

def map_PATIDS_to_split(PATIDS, PATIDS_train, PATIDS_val, PATIDS_test):
    if PATIDS in PATIDS_train:
        return 'train'
    elif PATIDS in PATIDS_val:
        return 'val'
    else:
        assert PATIDS in PATIDS_test
        return 'test'  

def getVocab(loc=VOCABULARY):
    loadpickle = None
    with open(loc, 'rb') as handle:
        loadpickle = pickle.load(handle)
    diags_label,procs_label,drugs_label,labs_label, Y_labels = loadpickle
    return (drugs_label+labs_label+procs_label+diags_label, Y_labels)

def dataSlice(X,y,weights,treatment,mask):
    return {'X':X, 'y':y,'W':treatment , 'weights':weights, 'mask':mask}

def getWANDBDir(run):
    return  MODEL_OUTPUTS + '{}/{}'.format(run.project,run.name)

def logWANDBRun(run):
    run_path = MODEL_OUTPUTS + '{}/{}'.format(run.project,run.name)
    pathlib.Path(run_path).mkdir(parents=True, exist_ok=True)
    file = open(run_path+'/config.json','w')
    file.write(str(run.config))
    file.close()

def parseDatasetNew(data, trt, all_labels, ratio=[0.5,0.60, 1],seed=999, dropEmpty=False, se = 'all', weight_control=100, excludePatids=[], keep_treatment_col=False):
    
    labels, y_labels = all_labels

    if excludePatids is not None:
        keepRows = (~np.in1d(data['PATIDS'], excludePatids)).nonzero()[0]
        data['PATIDS'] = data['PATIDS'][keepRows]
        data['DATESTAMPS'] = data['DATESTAMPS'][keepRows]
        data['W'] = data['W'][keepRows]
        data['X'] = data['X'][keepRows,:]
        data['Y'] = data['Y'][keepRows,:]

    for trt_column in [list(labels).index(x) for x in trt.split("_")]:
        if not keep_treatment_col:
             data['X'][:,trt_column] = 0

    PATIDS_train, PATIDS_val, PATIDS_test = split_data(data,ratio=ratio, seed=seed)
    print('Split dataset', len(PATIDS_train), len(PATIDS_val), len(PATIDS_test))

    df = pd.DataFrame({'PATIDS': data['PATIDS'], 'DATESTAMPS': data['DATESTAMPS']})
    df['split'] = df['PATIDS'].apply(lambda x: map_PATIDS_to_split(x, PATIDS_train, PATIDS_val, PATIDS_test))

    train_mask = df['split'].to_numpy() == 'train'
    val_mask = df['split'].to_numpy() == 'val'
    test_mask = df['split'].to_numpy() == 'test'
    
    X_train = data['X'][train_mask,:]
    y_train = data['Y'][train_mask,:]
    train_treatment = data['W'][train_mask]
    train_weight = data['W'][train_mask].copy()
    train_weight[train_weight==0]=weight_control
    
    X_val = data['X'][val_mask,:]
    y_val = data['Y'][val_mask,:]
    val_treatment = data['W'][val_mask]
    val_weight = data['W'][val_mask].copy()
    val_weight[val_weight==0]=weight_control

    X_test = data['X'][test_mask,:]
    y_test = data['Y'][test_mask,:]
    test_treatment = data['W'][test_mask]
    test_weight = data['W'][test_mask].copy()
    test_weight[test_weight==0]=weight_control
    
    if dropEmpty:
        
        drop_rows = pd.Series(X_train.sum(axis=0).ravel().tolist()[0]).reset_index()
        drop_rows.columns = ['row_ix','sum']
        drop_rows = list(drop_rows[drop_rows['sum']>0]['row_ix'])
        
        print('Dropping {} columns'.format(X_train.shape[1]-len(drop_rows)))
        
        X_train = X_train[:,drop_rows]
        X_val = X_val[:,drop_rows]
        X_test = X_test[:,drop_rows]
        

    subsets = {'train':dataSlice(X_train,y_train,train_weight,train_treatment,train_mask), 'val':dataSlice(X_val,y_val,val_weight,val_treatment,val_mask), 'test':dataSlice(X_test,y_test,test_weight,test_treatment,test_mask)}

    #Pdb().set_trace()

    data_hash = hashlib.md5(pickle.dumps(data)).hexdigest()
    split_hash = hashlib.md5((str(ratio) + str(seed)).encode('utf-8')).hexdigest()

    #data['data'] = data_
    data['subsets'] = subsets
    data['hash'] =  data_hash+"_"+split_hash   
    
    return data

def parseDataset(loc, trt, labels, ratio=[0.5,0.60, 1],seed=999, dropEmpty=False):
    data = pkl.load(open(loc, "rb"))
    X_all = scipy.sparse.csr.csr_matrix(scipy.sparse.hstack([data['X'][0], data['X'][1],data['X'][2],data['X'][3],data['X'][4]]))
    
    data['X'] = X_all
    data['weights'] = weights = 1.0/data['sample']

    trt_column = list(labels).index(trt)
    X_all[:,trt_column] = 0

    PATIDS_train, PATIDS_val, PATIDS_test = split_data(data,ratio=ratio, seed=seed)
    print('Split dataset', len(PATIDS_train), len(PATIDS_val), len(PATIDS_test))

    df = pd.DataFrame({'PATIDS': data['PATIDS'], 'DATESTAMPS': data['DATESTAMPS']})
    df['split'] = df['PATIDS'].apply(lambda x: map_PATIDS_to_split(x, PATIDS_train, PATIDS_val, PATIDS_test))

    train_mask = df['split'].to_numpy() == 'train'
    val_mask = df['split'].to_numpy() == 'val'
    test_mask = df['split'].to_numpy() == 'test'

    X_train = data['X'][train_mask,:]
    y_train = data['Y'][train_mask]
    train_weights = data['weights'][train_mask]
    train_treatment = data['W'][train_mask]
    
    X_val = data['X'][val_mask,:]
    y_val = data['Y'][val_mask]
    val_weights = data['weights'][val_mask]
    val_treatment = data['W'][val_mask]


    X_test = data['X'][test_mask,:]
    y_test = data['Y'][test_mask]
    test_weights = data['weights'][test_mask]
    test_treatment = data['W'][test_mask]
    
    if dropEmpty:
        drop_rows = pd.Series(X_train.sum(axis=0).ravel().tolist()[0]).reset_index()
        drop_rows.columns = ['row_ix','sum']
        drop_rows = list(drop_rows[drop_rows['sum']>0]['row_ix'])
        
        print('Dropping {} rows'.format(X_train.shape[1]-len(drop_rows)))
        
        X_train = X_train[:,drop_rows]
        X_val = X_val[:,drop_rows]
        X_test = X_test[:,drop_rows]
        


    subsets = {'train':dataSlice(X_train,y_train,train_weights,train_treatment,train_mask), 'val':dataSlice(X_val,y_val,val_weights,val_treatment,val_mask), 'test':dataSlice(X_test,y_test,test_weights,test_treatment,test_mask)}


    data_hash = hashlib.md5(open(loc,'rb').read()).hexdigest()
    split_hash = hashlib.md5((str(ratio) + str(seed)).encode('utf-8')).hexdigest()

    return data, subsets, data_hash+"_"+split_hash


def makeDir(path):
    try: 
        os.mkdir(path) 
    except OSError as error: 
        pass  

def run_model(model_name, dataset_hash, force=False):
    path = MODEL_OUTPUTS + dataset_hash
    makeDir(path)
    
    if force:
        return True

    return not os.path.isfile(MODEL_OUTPUTS + dataset_hash + '/' + model_name.replace(" ","_") + '.csv') 

def write_model(data, subset, prediction, model_name, dataset_hash):
    # make DIR if it doesnt exist
    path = MODEL_OUTPUTS + dataset_hash
    makeDir(path)

    model_output_loc = MODEL_OUTPUTS + dataset_hash + '/' + model_name.replace(" ","_") + '.csv'
    
    sample_id = np.array(list(range(0, data['X'].shape[0])))
    sample_id = sample_id[subset['test']['mask']]
    
    model_output = pd.DataFrame.from_dict({'sample_id':sample_id, 'prediction':prediction})
    model_output.to_csv(model_output_loc,index=False)


def collateModels(dataset_hash, models):
    path = MODEL_OUTPUTS + dataset_hash
    
    collect = []
    
    for model_name in models:
        model_output_loc = MODEL_OUTPUTS + dataset_hash + '/' + model_name.replace(" ","_") + '.csv'
        result = pd.read_csv(model_output_loc)
        result.columns = ['sample_id', model_name]
        collect.append(result)
    
    df = collect[0]
    for i in range(1, len(collect)):
        df = df.merge(collect[i], on='sample_id')
            
    return df

def computeRecall(subset,full,col='Y'):
    numerator = subset[col].sum()
    denom = full[col].sum()
    return float(numerator)/float(denom)
    

def recallAt(gt, prediction, percentile=0.99):
    eval_df = pd.DataFrame.from_dict({'pred':prediction, 'gt':gt}).sort_values(by='pred',ascending=False)
    num_top = int(len(eval_df) * (1-percentile))
    subset = eval_df.head(n=num_top)
    return computeRecall(subset, eval_df, col='gt')


