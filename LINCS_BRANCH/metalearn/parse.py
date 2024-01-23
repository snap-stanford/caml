""" Utility modules for task definition, loading, sampling"""
import numpy as np
import pandas as pd
import os 
import pickle
from pathlib import Path
import pdb
from causal_singletask.consts_and_utils import (getVocab, parseDatasetNew, flattenShards)
from causal_singletask.consts_and_utils import (DATA_LOCATION, MODEL_OUTPUTS, VOCABULARY, VOCAB_JSON, DATABASE_LOCATION)
from pathlib import Path
from os.path import exists
from sklearn.base import clone
import pickle as pkl
from sklearn.metrics import roc_auc_score
import json
import scipy
from pdb import set_trace as bp
import glob
from scipy.sparse import csr_matrix, load_npz
import hashlib
import pathlib
import os
import time
import fcntl
import shutil


# exact match is if we want the exact combination of drugs
# if it is false we return any rows containing one of the drugs 


def open_file_for_exclusive_writing(filename):
    """Opens the given file for exclusive writing.

    Returns None if the given file already exists and is nonempty,
    or if another process has opened the file for exclusive writing.

    This function can be used to distribute jobs among various processes.
    When a process wants to start working on a job, it should first open the
    output file for exclusive writing. This will only succeed if the file is
    empty and not opened by another process for exclusive writing. Jobs will
    thus work on disjoint sets of tasks.

    Note that flock() works over NFS, so this works even if the jobs are run
    on different machines. See https://manpages.debian.org/flock(2)
    """
    # Open in append mode to avoid truncating the file.
    f = open(filename, 'ab')
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('Another process has locked file:', filename)
        f.close()
        return None
    if f.tell() != 0:
        print('File is nonempty:', filename)
        f.close()
        return None
    return f


def getTreatmentRows(features, TASK_ID, exactMatch = True):
    vocab_json = json.loads(open(VOCAB_JSON,'r').read())
    trt_columns = [vocab_json['Drugs'].index(x) for x in TASK_ID.split("_")]

    if exactMatch:

        W = features[:,trt_columns[0]]
        select_rows = W.nonzero()[0]
        task_drugs = TASK_ID.count('_')+1
    

        if task_drugs > 1:
            for z in range(1,task_drugs):
                W2 = features[:,trt_columns[z]]
                select_rows_filter = W2.nonzero()[0]
                select_rows = np.intersect1d(select_rows, select_rows_filter)
    
        num_drugs = len(vocab_json['Drugs'])
        total_drugs = features[:,0:num_drugs].sum(axis=1)
        select_rows = np.intersect1d(select_rows, (total_drugs == task_drugs).nonzero()[0])

    else:
        select_rows = features[:,trt_columns].sum(axis=1).nonzero()[0]

    return select_rows


class QueryPatientsByDrug:
    """ Loads patient information (X, Y) for a given task"""

    def __init__(
        self,
        data_path: str = DATABASE_LOCATION,
        vocab_path: str = VOCAB_JSON
    ):
        self.data_path = data_path

    def __call__(self, TASK_ID, force=False):
        pathlib.Path(DATA_LOCATION+'/split_caches/').mkdir(parents=True, exist_ok=True) 

        TASK_ID.sort()
        TASK_ID = '_'.join(TASK_ID)
        task_hash = hashlib.md5((str(TASK_ID)).encode('utf-8')).hexdigest()
        filename = DATA_LOCATION+'/split_caches/'+task_hash+'.csv'

        if os.path.isfile(filename):
            print('Loading data for split {} from location {}'.format(TASK_ID, filename))
            patids = np.loadtxt(filename, delimiter='\n',dtype='<U15')  

        else:

            print('Querying patitds for split {} and will save to location {}'.format(TASK_ID, filename))
    
            drug_files = [x for x in list(glob.glob(self.data_path+'/drugs/*'))]
            drug_files.sort(key=lambda x:  int(x.split('{}_shard_'.format('drugs'))[1].split('.npz')[0])   )
            patid_files = [x for x in list(glob.glob(self.data_path+'/patids/*'))]
            patid_files.sort(key=lambda x:  int(x.split('{}_shard_'.format('patids'))[1].split('.npz')[0])   )
    
            patid_arrays = []
    
            for i in range(0,len(drug_files)):
                if  i%10 == 0:
                    print('{}/500 patids loaded'.format(i))
                X = load_npz(drug_files[i])
                P = np.load(patid_files[i])['DATA']
                matchRows = getTreatmentRows(X, TASK_ID, exactMatch=False)
                patid_arrays.append(np.unique(P[matchRows]))
    
            patids = np.unique(np.concatenate(patid_arrays))
            np.savetxt(filename, patids, delimiter='\n', fmt='%s')  

        return patids

class CellDataLoader:
    """ Loads patient information (X, Y) for a given task"""

    def __init__(
        self,
        data_path: str,
        processed_ext:str=''
    ):
        self.data_path = data_path
        self.processed_ext = processed_ext

    def _load_pickle(self, path: str):
        with open(path,'rb') as f:
            return pickle.load(f)
     
    def __call__(self, task_id: str,forceProcess=False,se='Pancytopenia',clipY=True, excludePatids=None, split='train', keep_treatment_col=False,include_support_in_val=False):
        data = {}
        for split in ['train','val','test']:
            data[split] = {}
            data[split]['X'] = np.random.rand(100,19221)
            data[split]['y'] = np.random.rand(100,978)
            data[split]['tau'] = np.random.rand(100,978)
            data[split]['W'] =  (np.random.rand(100) > 0.5).astype(int)

        return data





class PatientDataLoader:
    """ Loads patient information (X, Y) for a given task"""

    def __init__(
        self,
        data_path: str,
        processed_ext:str=''
    ):
        self.data_path = data_path
        self.processed_ext = processed_ext

    def _load_pickle(self, path: str):
        with open(path,'rb') as f:
            return pickle.load(f)


    def mergeTreatmentControl(self, trt_task, ctrl_task, TASK_ID):
        output = {}

        ### Filter out CONTROL patients who took the treatment drug   
        ###
        ###   

        X = ctrl_task['X']        
        select_rows = getTreatmentRows(X, TASK_ID, exactMatch=True)
        excludePatids = ctrl_task['PATIDS'][select_rows]
        keepRows = (~np.in1d(ctrl_task['PATIDS'], excludePatids)).nonzero()[0]

        ctrl_task['PATIDS'] = ctrl_task['PATIDS'][keepRows]
        ctrl_task['DATESTAMPS'] = ctrl_task['DATESTAMPS'][keepRows]
        ctrl_task['X'] = ctrl_task['X'][keepRows,:]
        ctrl_task['Y'] = ctrl_task['Y'][keepRows,:]

        output['PATIDS'] = np.concatenate( [trt_task['PATIDS'], ctrl_task['PATIDS']]  )
        output['DATESTAMPS'] = np.concatenate(  [trt_task['DATESTAMPS'], ctrl_task['DATESTAMPS']]  )
        output['X'] = scipy.sparse.csr_matrix(scipy.sparse.vstack([trt_task['X'], ctrl_task['X']]))
        output['Y'] = scipy.sparse.csr_matrix(scipy.sparse.vstack([trt_task['Y'], ctrl_task['Y']]))
        output['W'] = np.concatenate((np.ones(trt_task['DATESTAMPS'].shape) , np.zeros(ctrl_task['DATESTAMPS'].shape))).astype(int)
        output['task'] = TASK_ID

        return output

     
    def __call__(self, task_id: str,forceProcess=False,se='Pancytopenia',clipY=True, excludePatids=None, split='train', keep_treatment_col=False,include_support_in_val=False):

        #pdb.set_trace()
        hashExclude = ""
        if excludePatids is not None:
            hashExclude = hashlib.md5((str(sorted(excludePatids))).encode('utf-8')).hexdigest()[:5]

        s_string = ''
        if keep_treatment_col:
            s_string=s_string+'S'
        if include_support_in_val:
            s_string = s_string+'VALSUPPORT'

        path_str = self.data_path + "/tasks/{}/processed{}_{}_{}{}.pkl".format(task_id, self.processed_ext,split,hashExclude,s_string)

        #processed = Path(path_str)
        X_labels, Y_labels = getVocab()

        data = None

        ratio = None
        # TRAIN: 90 TRAIN / 10 VAL / 0 TEST
        if split == 'train':
            ratio = [0.9, 0.99, 1.0]
        # VAL: 0 TRAIN / 0 VAL / 100 TEST

        elif split == 'val':
            ratio = [0.01,0.02,1.0]

        # VAL: 50 TRAIN / 0 VAL / 50 TEST
        elif split == 'test':
            ratio = [0.50,0.501,1.0]

        # VAL: 50 TRAIN / 0 VAL / 50 TEST

        if split == 'val' and include_support_in_val==True:
            ratio = [0.50,0.501,1.0]

        outfile = open_file_for_exclusive_writing(path_str)
        loadSuccess = False

        if outfile is None:
            for i in range(0,20):
                print("Attempt {} to read data from {}".format(i,path_str))
                try:
                    with open(path_str, 'rb') as handle:
                        data = pickle.load(handle)
                    loadSuccess = True
                    break
                except:
                    time.sleep(15*1)

            if loadSuccess == False:
                print("I GAVE UP AND OVERWROTE THE FILE")
                print("Attempting to write data to {}".format(path_str))
                treatment_shards = flattenShards(task_id, data_path = self.data_path)
                control_shards = flattenShards('Meta{}Ctrl'.format(split) , data_path = self.data_path)
                merged_task = self.mergeTreatmentControl(treatment_shards, control_shards, task_id)
                data = parseDatasetNew(merged_task, task_id, all_labels=getVocab(), dropEmpty=False,se='all',ratio=ratio, excludePatids=excludePatids, keep_treatment_col = keep_treatment_col)
                with open(path_str, 'wb') as handle:
                    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        else:
            with outfile:
                print("Attempting to write data to {}".format(path_str))
                treatment_shards = flattenShards(task_id, data_path = self.data_path)
                control_shards = flattenShards('Meta{}Ctrl'.format(split) , data_path = self.data_path)
                merged_task = self.mergeTreatmentControl(treatment_shards, control_shards, task_id)
                data = parseDatasetNew(merged_task, task_id, all_labels=getVocab(), dropEmpty=False,se='all',ratio=ratio, excludePatids=excludePatids, keep_treatment_col = keep_treatment_col)
                with open(path_str+"_temp", 'wb') as handle:
                    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
                shutil.copyfileobj(open(path_str+"_temp", 'rb'), outfile)
                os.unlink(path_str+"_temp")

            try:
                os.chmod(path_str , 0o777)
            except:
                time.sleep(60*5)
                try: 
                    os.chmod(path_str , 0o777)
                except:
                    time.sleep(60*5)
                    os.chmod(path_str , 0o777)


            print('Writing pickle file for Task {} in location {}'.format(task_id, path_str))


        if se != 'all':
            data['Y'] = np.array(data['Y'][:,Y_labels.index(se)].todense().ravel().tolist()[0])
            data['subsets']['train']['y'] = np.array(data['subsets']['train']['y'][:,Y_labels.index(se)].todense().ravel().tolist()[0])
            data['subsets']['val']['y'] = np.array(data['subsets']['val']['y'][:,Y_labels.index(se)].todense().ravel().tolist()[0])
            data['subsets']['test']['y'] = np.array(data['subsets']['test']['y'][:,Y_labels.index(se)].todense().ravel().tolist()[0])

        if clipY:
            data['Y'] = data['Y'].clip(0,1)
            data['subsets']['train']['y'] = data['subsets']['train']['y'].clip(0,1)
            data['subsets']['val']['y'] = data['subsets']['val']['y'].clip(0,1)
            data['subsets']['test']['y'] = data['subsets']['test']['y'].clip(0,1)


        return data



class XLearnerM0:
    """Takes a dataset as input and returns a new dataset with tau nuisance parameters adjusted"""
    def __init__(
        self,
        m0,
    ):
        self.m0 = m0
    
    def __call__(self, data, forceRetrain=False):

        nuisance_dir = DATA_LOCATION +"{}/nuisance_models".format(data['task'])
        nuisance_model_file = nuisance_dir + '/{}.pkl'.format(data['hash'])


        Path(nuisance_dir).mkdir(parents=True, exist_ok=True)
        
        subsets = data['subsets']

        X = subsets['train']['X']
        y = subsets['train']['y']
        treatment = subsets['train']['W']        
        weights = subsets['train']['weights']
        mask = treatment.astype(bool)

        # Only train model if you need to 
        file_exists = exists(nuisance_model_file)
        model = None

        try:
            if file_exists and not forceRetrain:
                with open(nuisance_model_file, 'rb') as handle:
                    model = pickle.load(handle)
            else:
                model = clone(self.m0)
                model.fit(X[~mask,:], y[~mask], weights[~mask])
                with open(nuisance_model_file, 'wb') as handle:
                    pkl.dump(model, handle)

                try:
                    os.chmod(nuisance_model_file , 0o777)
                except:
                    time.sleep(60*5)
                    try: 
                        os.chmod(nuisance_model_file , 0o777)
                    except:
                        time.sleep(60*5)
                        os.chmod(nuisance_model_file , 0o777)


        except:
            model = clone(self.m0)
            model.fit(X[~mask,:], y[~mask], weights[~mask])
            with open(nuisance_model_file, 'wb') as handle:
                pkl.dump(model, handle)
                
                try:
                    os.chmod(nuisance_model_file , 0o777)
                except:
                    time.sleep(60*5)
                    try: 
                        os.chmod(nuisance_model_file , 0o777)
                    except:
                        time.sleep(60*5)
                        os.chmod(nuisance_model_file , 0o777)


        for split in ['train','val','test']:
            
            # Get all relevant data
            X = subsets[split]['X']
            y = subsets[split]['y']
            treatment = subsets[split]['W']        
            weights = subsets[split]['weights']
            mask = treatment.astype(bool)


            # Evaluate train model on test data
            if split == 'val' and np.sum(y[~mask])>0:
                y_hat = model.predict_proba(X[~mask,:])[:,1]
                
                if 'memory' not in data:
                    data['memory'] = {}
                data['memory']['m0_rocauc'] = roc_auc_score(y[~mask], y_hat)
                
                print("ROC AUC Score {}".format( data['memory']['m0_rocauc'] ))           
            
            # Create tau model
            subsets[split]['tau'] = subsets[split]['y'] - model.predict_proba(X)[:,1]
            
        return data


