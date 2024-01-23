import sys
import os
from sklearn.base import clone
import argparse 
from inspect import signature
import os
import gzip
import json
import scipy
import random
import math
import numpy as np
import pickle as pkl
import pandas as pd
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
from sklearn.model_selection import cross_val_predict, KFold, train_test_split
from scipy.sparse import hstack
import pickle
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from IPython.core.debugger import Pdb;
from IPython import embed
import sys 
from typing import Union
from scipy.sparse import csr_matrix, coo_matrix
from causal_singletask.consts_and_utils import *
import pdb; 
import causal_singletask.models
from causal_singletask import models
from causal_singletask.torch_utils import BestValueLogger, EarlyStopping
from metalearn.parse import PatientDataLoader
from sklearn.metrics import roc_auc_score
from causal_singletask.consts_and_utils import (getVocab, parseDatasetNew, flattenShards)
from causal_singletask.consts_and_utils import (DATA_LOCATION, MODEL_OUTPUTS, VOCABULARY, collateModels)
from pathlib import Path
from os.path import exists
import time

def countPositiveLabels(task):
    eval_df = pd.DataFrame({'y':task['Y'], 'treatment': task['W']})
    eval_df = eval_df[eval_df['treatment']==1]
    return eval_df['y'].sum()

def treatmentRecall(subset, rankings, thresholds = [0.001,0.002]):
    
    eval_df = pd.DataFrame({'y':subset['y'], 'score': rankings, 'treatment': subset['W']})
    eval_df = eval_df[eval_df['treatment']==1]
    
    recalls = []
    
    for threshold in thresholds:
        num_top = int(len(eval_df) * threshold)
        print("Computing for Top {}".format(str(1.0-threshold)))
        
        subset = eval_df.sort_values(by='score',ascending=False).head(n=num_top)
        numerator = float(subset['y'].sum())
        denom = float(eval_df['y'].sum())
        recalls.append(numerator/denom)
        
    table = {'threshold':[(1.0-x) for x in thresholds], 'recall':recalls}

    return table

def treatmentPrecision(subset, rankings, thresholds = [0.001,0.002]):
    
    eval_df = pd.DataFrame({'y':subset['y'], 'score': rankings, 'treatment': subset['W']})
    eval_df = eval_df[eval_df['treatment']==1]
    
    recalls = []
    
    for threshold in thresholds:
        num_top = int(len(eval_df) * threshold)
        print("Computing for Top {}".format(str(1.0-threshold)))
        
        subset = eval_df.sort_values(by='score',ascending=False).head(n=num_top)
        numerator = float(subset['y'].mean())
        recalls.append(numerator)
        
    table = {'threshold':[(1.0-x) for x in thresholds], 'precision':recalls}

    return table

def treatmentROCAUC(subset, rankings):
    
    eval_df = pd.DataFrame({'y':subset['y'], 'score': rankings, 'treatment': subset['W']})
    eval_df = eval_df[eval_df['treatment']==1]
    

    return roc_auc_score(list(eval_df['y']), list(eval_df['score']))


def treatmentChoose2(subset, rankings):

    eval_df = pd.DataFrame({'y':subset['y'], 'score': rankings, 'treatment': subset['W']})
    eval_df = eval_df[eval_df['treatment']==1]

    pos = eval_df[eval_df['y']==1]
    pos['row_number'] = np.arange(len(pos))
    neg = eval_df[eval_df['y']==0].sample(n=len(pos))
    neg['row_number'] = np.arange(len(neg))
    eval_df = pd.concat([pos,neg]).sort_values(by=['row_number'])
    
    correctPure = 0
    correct = 0

    for row_number in np.arange(len(neg)):
        eval_df_ = eval_df[eval_df['row_number']==row_number].sort_values(by='y').reset_index()
        right=False
        
        if eval_df_.iloc[0]['score']  < eval_df_.iloc[1]['score']:
            right=True
            correctPure+=1    
            correct+=1    

        elif eval_df_.iloc[0]['score']  == eval_df_.iloc[1]['score']:
            correct += 0.5

    return {'choose2':[float(correct)/float(len(neg)), float(correctPure)/float(len(neg))], 'threshold':['vanilla','pure']}



def addLogDict(dic,table,task,split='Test',metric_name='recall',higherIsBetter=True):

    directionality = '↑'
    if not higherIsBetter:
        directionality = '↓'

    for i, thresh in enumerate(table['threshold']):
        name = '{} {} @ {} ({}) {}'.format(split,metric_name,thresh,task,directionality)
        value = table[metric_name][i]
        dic[name] = value
    
def treatmentRATE(subset, rankings, model, thresholds = [0.001,0.002], forceRetrain=False):
    
    eval_dir = DATA_LOCATION +"/evals"
    Path(eval_dir).mkdir(parents=True, exist_ok=True)
    data_hash = hashlib.md5(pickle.dumps(subset)).hexdigest()
    model_hash = hashlib.md5(pickle.dumps(model)).hexdigest()
    rate_score_file = eval_dir + '/{}_{}.csv'.format(data_hash,model_hash)
    rate_model_path = Path(rate_score_file)

    df_DR = None

    if not rate_model_path.is_file() or forceRetrain:
        NFOLD = 5
        CV = KFold(n_splits=NFOLD, shuffle=True, random_state=42)
        treat = subset['W']
        m0_learner = clone(model)  #RandomForestClassifier(random_state=0, n_estimators=150, max_depth=30,n_jobs=75,verbose=True)
        m0_hat_control = cross_val_predict(m0_learner, subset['X'][~treat.astype(bool),:], subset['y'][~treat.astype(bool)], cv=CV, n_jobs=32, method='predict_proba', fit_params={'sample_weight': subset['weights'][~treat.astype(bool)] })[:,1]
        m0_learner.fit(subset['X'][~treat.astype(bool),:], subset['y'][~treat.astype(bool)], sample_weight=subset['weights'][~treat.astype(bool)])
        m0_hat_treatment = m0_learner.predict_proba(subset['X'][treat.astype(bool),:])[:,1]
        m0 = np.hstack([m0_hat_control,m0_hat_treatment])
        
        m1_learner = clone(model)   #. RandomForestClassifier(random_state=0, n_estimators=150, max_depth=30,n_jobs=75,verbose=True)
        m1_hat_treatment = cross_val_predict(m1_learner, subset['X'][treat.astype(bool),:], subset['y'][treat.astype(bool)], cv=CV, n_jobs=32, method='predict_proba', fit_params={'sample_weight': subset['weights'][treat.astype(bool)] })[:,1]
        m1_learner.fit(subset['X'][treat.astype(bool),:], subset['y'][treat.astype(bool)], sample_weight=subset['weights'][treat.astype(bool)])
        m1_hat_control = m1_learner.predict_proba(subset['X'][~treat.astype(bool),:])[:,1]
        m1 = np.hstack([m1_hat_control,m1_hat_treatment])
        
        indices_dr = np.arange(0,len(subset['y']))
        indices = np.hstack([indices_dr[~treat.astype(bool)], indices_dr[treat.astype(bool)]])
        
        df_DR = pd.DataFrame.from_dict({'m0':m0,'m1':m1,'indices':indices})
        df_DR['tau'] = df_DR['m1']-df_DR['m0']
        df_DR.sort_values(by='indices',ascending=True,inplace=True)
        df_DR.reset_index(inplace=True)
        
        
        mhat_w = np.where(treat==0,
                       np.array(df_DR['m0']),
                       np.array(df_DR['m1']))
        
        g_standard = clone(model)
        ehat_standard = cross_val_predict(g_standard, subset['X'], subset['W'], cv=CV, n_jobs=-1, method='predict_proba', fit_params={'sample_weight': subset['weights'] })[:, 1]
        
        df_DR['residual'] = subset['y'] - mhat_w
        df_DR['adjustor'] = ((treat - ehat_standard)/(np.multiply(ehat_standard,1-ehat_standard)))
        df_DR['DR_SCORE'] = df_DR['tau'] + df_DR['adjustor'] * df_DR['residual']

        rate_model_path2 = Path(rate_score_file)
        if not rate_model_path2.is_file():
            df_DR.to_csv(rate_score_file)

            try:
                os.chmod(rate_score_file , 0o777)
            except:
                time.sleep(60*5)
                try: 
                    os.chmod(rate_score_file , 0o777)
                except:
                    time.sleep(60*5)
                    os.chmod(rate_score_file , 0o777)


    else:
        df_DR = pd.read_csv(rate_score_file)


    eval_df = pd.DataFrame({'y': df_DR['DR_SCORE'], 'score': rankings, 'treatment': subset['W']})
    eval_df = eval_df[eval_df['treatment']==1]
    
    rates = []
    
    for threshold in thresholds:
        num_top = int(len(eval_df) * threshold)
        print("Computing for Top {}".format(str(1.0-threshold)))
        subset = eval_df.sort_values(by='score',ascending=False).head(n=num_top)
        top = subset['y'].mean()
        all_pat = eval_df['y'].mean()
        
        rates.append(top - all_pat)
        
        
    table = {'threshold':[(1.0-x) for x in thresholds], 'rate':rates}
    
    return table
