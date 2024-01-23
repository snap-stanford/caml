#!/usr/bin/env python
# coding: utf-8

# In[1]:



import gzip
import json
import numpy as np
import scipy
import pickle
import pandas as pd
from distributed import Client
import time
import pyarrow as pa
from datetime import datetime
from multiprocessing import Process
import multiprocessing as mp
import itertools
import glob
from scipy.sparse import csr_matrix, load_npz
import dask
import dask.dataframe as dd
from distributed import Client
import random
pd.set_option('display.max_rows', 3000)
import pathlib
from IPython.core.debugger import Pdb
import sys
import os
    
SERVER_INDEX = int(sys.argv[1])
NUM_SERVERS = 1
SE_WINDOWS = [90,30,7] #days after the drug was taken to monitor the side effect
DB_LOCATION = '/lfs/trinity/0/anonn/segregation/checkpoint_4/reordered/'
DATA_DIR = '/lfs/trinity/0/anonn/segregation/exports'
DRUG_TASKS_LOCATION = '/lfs/trinity/0/anonn/segregation/task_splits.csv'



processes = []
running = []
i=0
maxP=80

def flatten(t):
    return [item for sublist in t for item in sublist]


def days_between(d1, d2):
    d1 = datetime.strptime(d1, "%Y-%m-%d")
    d2 = datetime.strptime(d2, "%Y-%m-%d")
    return abs((d2 - d1).days)

def setupParallel():
    maxP = 80
    running = []
    processes = []
    i=0

def runParallel():
    for i in range(0,maxP):
        
        if len(processes)==0:
            break
        
        process = processes.pop(0)
        process.start()
        running.append(process)

    while True:
        if (len(processes) == 0 and len(running) == 0):
            break
        for i in reversed(range(0,len(running))):
            if (not running[i].is_alive()):
                running.pop(i)
                if (len(processes) > 0):
                    process = processes.pop(0)
                    process.start()
                    running.append(process)
        time.sleep(15)

def loadColumns(DB_LOCATION,column_name,start_row=0,end_row=-1,sparse=True):
    files = [x for x in list(glob.glob(DB_LOCATION+column_name+'/*'))]
    files.sort(key=lambda x:  int(x.split('{}_shard_'.format(column_name))[1].split('.npz')[0])   )

    accumulate_loaded = []
    
    if end_row == -1:
        end_row=len(files)
            
    for i in files[start_row:end_row]:
        if sparse:
            loaded = load_npz(i)
        else:
            loaded = np.load(i)['DATA']
        print('Loading {}'.format(i))
        accumulate_loaded.append(loaded)
        
    if sparse:
        return scipy.sparse.vstack(accumulate_loaded)
    else:
        return np.concatenate(accumulate_loaded)
    
    
def getStartEndPATIDRows(PATID_LIST):
    #Pdb().set_trace()                                                                                                                                                                                                                        

    start_rows = (np.roll(PATID_LIST,shift=+1) != PATID_LIST).nonzero()[0]  

    # if there is only 1 patient in the entire shard
    if len(start_rows) == 0:
        start_end = pd.DataFrame.from_dict({'PATID':PATID_LIST[0],'start':[0],'end':[len(PATID_LIST)-1]})
        return start_end

    end_rows = np.roll(start_rows,shift=-1)-1
    end_rows[-1] = len(PATID_LIST)-1
    start_end = pd.DataFrame.from_dict({'PATID':PATID_LIST[start_rows],'start':start_rows,'end':end_rows})
    return start_end


def processShard(X, Y, P, D, I):
    print('processing shard with P length {}'.format(len(P)))

    if 'Ctrl' in TASK_ID:
        # Rows where a random drug was taken (control)
        drug_taken = X[:,0:Drugs.shape[1]].sum(axis=1).nonzero()[0]
        rows_drugs_taken = pd.DataFrame.from_dict({'patid':P[drug_taken],'rowid':drug_taken})
        select_rows2 = np.array(rows_drugs_taken.sample(frac=1).drop_duplicates(subset=['patid'],keep='first')['rowid'])
        drug_day = np.sort( select_rows2)

    else:
            
        # Rows where methotrexate was ONLY DRUG taken (treatment)
        trt_columns = [vocab_json['Drugs'].index(x) for x in TASK_ID.split("_")]
    
        W = X[:,trt_columns[0]]
        select_rows = W.nonzero()[0]
    
        task_drugs = TASK_ID.count('_')+1
    
        if task_drugs > 1:
            for z in range(1,task_drugs):
                W2 = X[:,trt_columns[z]]
                select_rows_filter = W2.nonzero()[0]
                select_rows = np.intersect1d(select_rows, select_rows_filter)
    
        num_drugs = len(vocab_json['Drugs'])
        total_drugs = X[:,0:num_drugs].sum(axis=1)
        ## Keep only the days where you received no other drugs besides the treatment drug 
        select_rows = np.intersect1d(select_rows, (total_drugs == task_drugs).nonzero()[0])
        
        rows_drugs_taken = pd.DataFrame.from_dict({'patid':P[select_rows],'rowid':select_rows}).drop_duplicates(subset=['patid'],keep='first')
        select_rows = np.array(rows_drugs_taken['rowid'])
        select_pats = np.array(rows_drugs_taken['patid'])
        drug_day = select_rows

    if drug_day.size == 0:
        return

    start_end = getStartEndPATIDRows(P)
    start_rows = list(start_end['start'])
    end_rows = list(start_end['end'])
    
    # Generate datestamps and final patid, ds, etc.
    D_str = np.copy(D)
    patid_final = P[drug_day]
    ds_final = D_str[drug_day]
    # W_final = np.array(W[drug_day,:].todense().ravel().tolist()[0])
    
    D = pd.Series(D,dtype=None)
    D = pd.to_datetime(D).view(int)
    D = (D // (10**9 * 86400))
    D = np.array(D)
    
    X_0 = []
    X_1 = []
    X_7 = []
    X_30 = []
    X_90 = []
    X_365 = []
    X_rest = []
    Y_90 = []
    Y_30 = []
    Y_7 = []

    print('start generating shard {}'.format(I))
    j = 0
    for i in range(0,len(drug_day)):
        DRUG_I = drug_day[i] 
        START_I = start_rows[j]
        END_I = end_rows[j]

        while not (START_I <= DRUG_I and DRUG_I <= END_I):
            j+=1
            START_I = start_rows[j]
            END_I = end_rows[j]
     
        patDS = D[START_I:DRUG_I+1]

        DAY1 = np.searchsorted(patDS, D[DRUG_I] - 1) + START_I
        DAY7 = np.searchsorted(patDS, D[DRUG_I] - 7) + START_I
        DAY30 = np.searchsorted(patDS, D[DRUG_I] - 30) + START_I
        DAY90 = np.searchsorted(patDS, D[DRUG_I] - 90) + START_I
        DAY365 = np.searchsorted(patDS, D[DRUG_I] - 365)  + START_I
          
        X_0.append(X[DRUG_I: DRUG_I+1].sum(axis=0))
        X_1.append(X[DAY1: DRUG_I].sum(axis=0))
        X_7.append(X[DAY7: DAY1].sum(axis=0))
        X_30.append(X[DAY30: DAY7].sum(axis=0))
        X_90.append(X[DAY90: DAY30].sum(axis=0))
        X_365.append(X[DAY365: DAY90].sum(axis=0))
        X_rest.append(X[START_I: DAY365].sum(axis=0))
        
        patDS = D[START_I:END_I]
        DAY90_fut = np.searchsorted(patDS, D[DRUG_I] + 90) + START_I
        Y_90.append(Y[DRUG_I: DAY90_fut].sum(axis=0)) 
        DAY30_fut = np.searchsorted(patDS, D[DRUG_I] + 30) + START_I
        Y_30.append(Y[DRUG_I: DAY30_fut].sum(axis=0)) 
        DAY7_fut = np.searchsorted(patDS, D[DRUG_I] + 7) + START_I
        Y_7.append(Y[DRUG_I: DAY7_fut].sum(axis=0)) 
      
    X_0_all =  scipy.sparse.vstack([scipy.sparse.csr_matrix(x) for x in X_0])
    X_1_all =  scipy.sparse.vstack([scipy.sparse.csr_matrix(x) for x in X_1])
    X_7_all =  scipy.sparse.vstack([scipy.sparse.csr_matrix(x) for x in X_7])
    X_30_all =  scipy.sparse.vstack([scipy.sparse.csr_matrix(x) for x in X_30])
    X_90_all =  scipy.sparse.vstack([scipy.sparse.csr_matrix(x) for x in X_90])
    X_365_all =  scipy.sparse.vstack([scipy.sparse.csr_matrix(x) for x in X_365])
    X_rest_all =  scipy.sparse.vstack([scipy.sparse.csr_matrix(x) for x in X_rest])
    Y_90_all =  scipy.sparse.vstack([scipy.sparse.csr_matrix(x) for x in Y_90])
    Y_30_all =  scipy.sparse.vstack([scipy.sparse.csr_matrix(x) for x in Y_30])
    Y_7_all =  scipy.sparse.vstack([scipy.sparse.csr_matrix(x) for x in Y_7])

    data={'X':[X_0_all, X_1_all, X_7_all, X_30_all, X_90_all, X_365_all, X_rest_all],'Y':[Y_90_all, Y_30_all, Y_7_all], 'W': 'Ctrl' not in TASK_ID,'DS':ds_final, 'PATID':patid_final}
    
    with open(DATA_DIR+'/tasks/'+TASK_ID+'/shards/{}.pkl'.format(I), 'wb') as handle:
        pickle.dump( data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    

def generateTask(TASK_ID):
    
    client = Client(n_workers=15)

    if 'Ctrl' not in TASK_ID:
        df_task_patids = dd.read_csv('/lfs/trinity/0/anonn/segregation/checkpoint_4/CausalMetaBigQuery/patient_ids/task_patid_{}drug*.csv'.format(  TASK_ID.count('_')+1 ))

    
        print('get relevant patients')
        RELEVANT_PATIENTS = list(df_task_patids[df_task_patids['TASK_ID']==TASK_ID]['Patid'].compute())

        print('start query ')
        #all patients who took drug 
        start_end_relevant = start_end[start_end['PATID'].isin(RELEVANT_PATIENTS)]
        print('start query1 ')
        if len(start_end_relevant) > 1000000:
            start_end_relevant = start_end_relevant.sample(n=1000000, replace=False)
        
        print('start sample2 ')
        # patients who did take the drug AND are in the treatment group (TODO: remove patients in unseen val/test split)
        start_end_trt = start_end_relevant[start_end_relevant['PATID'].isin(PATIDS_trt)]
        start_end_trt['group'] = 'trt'

        start_end_relevant = start_end_trt.sort_values(by='start',ascending=True)

    else:
        start_end_ctrl = start_end[(start_end['PATID'].isin(cntrl_groups[TASK_ID]))]
        start_end_ctrl['group'] = 'ctrl'
        start_end_relevant = start_end_ctrl.sort_values(by='start',ascending=True)
    
    array1 = np.array(list(start_end_relevant['start']))#[:49]
    array2 = np.array(list(start_end_relevant['end']))#[:49]
    first, last = array1.min(), array2.max()
    array3 = np.zeros(last-first+1, dtype='i1')
    array3[array1-first] = 1
    array3[array2-first] = -1
    array3 = np.flatnonzero(array3.cumsum())+first
    array3 = np.sort(np.concatenate((array3, array2)))
    
    Procedures_ = Procedures[array3,:]
    Diagnoses_ = Diagnoses[array3,:]
    Drugs_ = Drugs[array3,:]
    Labs_ = Labs[array3,:]
    Linked_Side_Effects_ = Linked_Side_Effects[array3,:]
    PATID_ = PATID[array3]
    DS_ = DS[array3]
    
    X_ = scipy.sparse.csr_matrix(scipy.sparse.hstack([ Drugs_, Labs_, Procedures_, Diagnoses_]))
    Y_ = Linked_Side_Effects_
    
    start_end_task = getStartEndPATIDRows(PATID_)
    
    SHARDS = 80
    idx = np.round(np.linspace(0, len(start_end_task) - 1, SHARDS+1)).astype(int)
    
    
    maxP2 = 80
    running = []
    
    for i_shard in range(1,len(idx)):
            
        while True:
            if len(running) < maxP2:
                break
            else:
                for j in reversed(range(0,len(running))):
                    if (not running[j].is_alive()):
                        running.pop(j)
            time.sleep(1)
                
        print(i_shard)
            
        print("remaining shard {}".format(i_shard))
        
        shard_start = idx[i_shard-1]
        shard_end = idx[i_shard] 
        
        
        row_start = start_end_task.iloc[shard_start]['start']
        row_end = start_end_task.iloc[shard_end]['start']
    
        if i_shard == len(idx)-1:
            row_end = start_end_task.iloc[shard_end]['end']
    
        X_shard =           X_[row_start:row_end,:]
        Y_shard =           Y_[row_start:row_end,:]
        pat_id_shard =     PATID_[row_start:row_end]    
        ds_shard =         DS_[row_start:row_end]
        
        if len(pat_id_shard) == 0:
            continue        

        #processShard(X_shard,Y_shard, pat_id_shard, ds_shard, i_shard)

        p = Process(target=processShard, args=(X_shard,Y_shard, pat_id_shard, ds_shard, i_shard))
        p.start()
        running.append(p)         
        

if __name__ == '__main__':
    
    #mp.set_start_method('forkserver')
    
    print("Processing on server index {}".format(SERVER_INDEX))
    
    Diagnoses = loadColumns(DB_LOCATION,'diagnoses')
    Procedures = loadColumns(DB_LOCATION,'procedures')
    Drugs = loadColumns(DB_LOCATION,'drugs')
    Labs = loadColumns(DB_LOCATION,'labs')
    Linked_Side_Effects = loadColumns(DB_LOCATION,'linked_side_effects')
    PATID = loadColumns(DB_LOCATION,'patids',sparse=False)
    DS = loadColumns(DB_LOCATION,'datestamps',sparse=False)
    
    VOCAB = '/lfs/trinity/0/anonn/segregation/checkpoint_4/CausalMetaBigQuery/vocab.json'
    vocab_json = json.loads(open(VOCAB,'r').read())
    

    



    no_drugs = set(pd.read_csv('/lfs/trinity/0/anonn/segregation/checkpoint_4/CausalMetaBigQuery/no_drugs.csv')['PATID'])
    
    start_end = getStartEndPATIDRows(PATID)   
    start_end.to_csv('/lfs/trinity/0/anonn/segregation/checkpoint_4/start_end.csv')
    start_end = pd.read_csv('/lfs/trinity/0/anonn/segregation/checkpoint_4/start_end.csv')
    

    def split_data():
        print('created pat vocab')
        PATIDS_vocab = set(start_end['PATID'])
        print(len(PATIDS_vocab))
        print('removing items')
        PATIDS_vocab = PATIDS_vocab.difference(no_drugs)
        #print('removing items')
        #print(len(PATIDS_vocab))
        #for i in set(no_drugs['PATID']):
        #    if str(i) in PATIDS_vocab:
        #        PATIDS_vocab.remove(str(i))
        print(len(PATIDS_vocab))
        print('done removing items')
        PATIDS_vocab = sorted(PATIDS_vocab)
        random.seed(4321)
        random.shuffle(PATIDS_vocab)
        ratio = [0.02,0.03,0.05,1.0]
        nums  = [int(len(PATIDS_vocab) * r) for r in ratio]
        PATIDS_ctrl_train = set(PATIDS_vocab[:nums[0]])
        PATIDS_ctrl_val   = set(PATIDS_vocab[nums[0]:nums[1]])
        PATIDS_ctrl_test  = set(PATIDS_vocab[nums[1]:nums[2]])
        PATIDS_trt  =         set(PATIDS_vocab[nums[2]:])
    
        return PATIDS_ctrl_train, PATIDS_ctrl_val, PATIDS_ctrl_test, PATIDS_trt
    
    
    PATIDS_ctrl_train, PATIDS_ctrl_val, PATIDS_ctrl_test, PATIDS_trt = split_data()
    len(PATIDS_ctrl_train), len(PATIDS_ctrl_val), len(PATIDS_ctrl_test), len(PATIDS_trt)
    
    cntrl_groups = {
        'MetatrainCtrl':PATIDS_ctrl_train,
        'MetavalCtrl':PATIDS_ctrl_val,
        'MetatestCtrl':PATIDS_ctrl_test
    }


    df_drug_tasks =  pd.concat(np.array_split(pd.read_csv(DRUG_TASKS_LOCATION), NUM_SERVERS)[SERVER_INDEX:])
    






    
    maxP = 20
    running2 = []

    print("Processing {} tasks".format(len(df_drug_tasks)))

    for ix, row in df_drug_tasks.iterrows():

    
        TASK_ID = row['Drugs']
        
        
        isExist = os.path.exists(DATA_DIR+'/tasks/'+TASK_ID+'/shards')
    
        if isExist:
            print("Task for {} already exists".format(TASK_ID))
            continue
       
        print("Created Task for {}".format(TASK_ID))
        
        pathlib.Path(DATA_DIR+'/tasks/'+TASK_ID+'/shards').mkdir(parents=True, exist_ok=True) 

         
        while True:
            if len(running2) < maxP:
                break
            else:
                for j in reversed(range(0,len(running2))):
                    if (not running2[j].is_alive()):
                        running2.pop(j)
            time.sleep(1)
        
        #generateTask(TASK_ID)
        p = Process(target=generateTask, args=(TASK_ID, ))
   
        p.start()
        running2.append(p)  

    for i in range(0,10000):
        print('sleeping...')
        time.sleep(1)

        
