import glob
import pickle
import scipy
import numpy as np
import os
import pathlib
from joblib import Parallel, delayed
from IPython import embed
import fire
from tqdm import tqdm
import gc
from time import time
	
class ParaLoader:
    """Util class to load pickle in parallel"""
    def __init__(self, n_jobs=30, n_chunks=30):
        self.n_jobs = n_jobs
        self.n_chunks = n_chunks

    def load_pickle(self, fname):
	    with open(fname, 'rb') as f:
	        return pickle.load(f)

    def load_files(self, files):
        result = [self.load_pickle(f) for f in files]
        gc.collect()
        return result

    def get_chunk_indices(self, x):
        n_rows = len(x) 
        chunk_size = int(np.floor(n_rows / self.n_chunks))
        chunk_indices = np.arange(0, n_rows, chunk_size, dtype=np.int64)
        start = chunk_indices
        end = chunk_indices + chunk_size
        if chunk_indices[-1] == n_rows:
            start = start[:-1]
            end = end[:-1]
        return start, end

    def __call__(self, files):
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self.load_files)(
                files[start: end] 
            ) for start, end in zip(
                *self.get_chunk_indices(
                    files 
                )
            )
        )
        # unpack list of list:
        return [inner for outer in results for inner in outer]

def check_reordering(x_old, x_new, order, n_tests=100):
    """check random (non-zero) rows whether they are identical between x_new[i] and x_old[order[i]]"""
    test_ids = np.random.choice(x_old.shape[0], n_tests)
    failed=False
    for i in test_ids:
        x_n = x_new[i] 
        x_o = x_old[order[i]] 
        if x_n.sum() + x_o.sum() > 0: #non trivial test
            x_n = x_n.todense()
            x_o = x_o.todense()
            try: 
                assert (x_n == x_o).sum() / x_n.shape[1] == 1
                #print(f'Test success for row {i}')
            except: 
                print(f'Failed to match row {i}')
                failed=True
    if not failed:
        print(f'Test passed.')

def check_reordering_array(x_old, x_new, order, n_tests=100):
    """check random (non-zero) rows whether they are identical between x_new[i] and x_old[order[i]]"""
    failed=False
    test_ids = np.random.choice(x_old.shape[0], n_tests)
    for i in test_ids:
        x_n = x_new[i] 
        x_o = x_old[order[i]] 
        try: 
            assert x_n == x_o 
            #print(f'Test success for row {i}')
        except: 
            print(f'Failed to match row {i}')
            failed=True
    if not failed:
        print(f'Test passed.')

def flatten(t):
    return [item for sublist in t for item in sublist]

def get_chunk_indices(x, n_chunks=100):
    n_rows = x.shape[0] 
    chunk_size = int(np.floor(n_rows / n_chunks))
    chunk_indices = np.arange(0, n_rows, chunk_size, dtype=np.int64)
    start = chunk_indices
    end = chunk_indices + chunk_size
    if chunk_indices[-1] == n_rows:
        start = start[:-1]
        end = end[:-1]
    return start, end
def get_chunk(x, start, end):
    """ extract chunk from matrix from start to end row"""
    return x[start:end, :]

def write_shard(x, fpath, iteration):
    fname = fpath.format(iteration)
    scipy.sparse.save_npz( fname, x, compressed=False)

def process_in_parallel(x, name, out_path='../../data/reordered', n_jobs=100, n_chunks=400):
    """ x: sparse scipy matrix
        name: name of dir and filename (e.g. procedures)
        out_path: path to output files
    """
    out_path = os.path.join(
        out_path, name
    )
    os.makedirs(
        out_path, 
        exist_ok = True
    )
    fpath = os.path.join(
        out_path, 
        name + '_shard_{:04}.npz'
    )
    _ = Parallel(n_jobs=n_jobs)(delayed(write_shard)(
        get_chunk(x, start, end), 
        fpath,
        i
    ) for i, (start, end) in enumerate(zip(*get_chunk_indices(x, n_chunks=n_chunks))))

def write_array_shard(x, fpath, iteration):
    fname = fpath.format(iteration)
    np.savez(fname, DATA=x)

def process_array_in_parallel(x, name, out_path='../../data/reordered', n_jobs=100, n_chunks=400):
    """ x: array of ids or datestamps
        name: name of dir and filename (e.g. patid)
        out_path: path to output files
    """
    out_path = os.path.join(
        out_path, name
    )
    os.makedirs(
        out_path, 
        exist_ok = True
    )
    fpath = os.path.join(
        out_path, 
        name + '_shard_{:04}.npz'
    )
    _ = Parallel(n_jobs=n_jobs)(delayed(write_array_shard)(
        x[start:end], 
        fpath,
        i
    ) for i, (start, end) in enumerate(zip(*get_chunk_indices(x, n_chunks=n_chunks))))


def add_in_parallel(x1, x2, n_jobs=300, n_chunks=500):
    def get_chunk_indices(x, n_chunks):
        n_rows = x.shape[0] 
        chunk_size = int(np.floor(n_rows / n_chunks))
        chunk_indices = np.arange(0, n_rows, chunk_size, dtype=np.int64)
        start = chunk_indices
        end = chunk_indices + chunk_size
        if chunk_indices[-1] == n_rows:
            start = start[:-1]
            end = end[:-1]
        return start, end

    def pairwise_add(x1, x2):
        return np.core.defchararray.add(x1, x2)

    results = Parallel(n_jobs=n_jobs)(delayed(pairwise_add)(
        x1[start:end], 
        x2[start:end] 
    ) for i, (start, end) in enumerate(zip(*get_chunk_indices(x1, n_chunks=n_chunks))))
    return np.concatenate(results)


def log_step(name, start_time):
    print(f'{name} took {time() - start_time:.2f} seconds.')

def main(
    path='../../../anon/ClaimsCausalMeta',
    n_jobs=20,
    n_chunks=500
):

    # get files: 
    files = [x for x in list(glob.glob(path +"/BigQuery/all_sparse_matrix/*")) if '.gz' not in x]
    files.sort(key=lambda x:  int(x.split('days_of_service_')[1].split('.json')[0])   )
   	
    loader = ParaLoader(n_jobs=n_jobs, n_chunks=n_chunks)
    t_start = time()
    shards = loader(files)

    #print(f'Completed loading shards, took {time() - t_start:.2f} sec.')
    log_step('Loading shards', t_start)

    ## Load raw data:
    #shards = []

    #for i in range(len(files[:3])):
    #    print(i)
    #    with open(files[i], 'rb') as handle:
    #        loadpickle = pickle.load(handle)
    #        shards.append(loadpickle
    #)
    t_stacking = time() 
    print('Stacking shards ..')         
    # STACK shards:
    Procedures = scipy.sparse.vstack(x['Procedures'] for x in shards)
    Diagnoses = scipy.sparse.vstack(x['Diagnoses'] for x in shards)
    Drugs = scipy.sparse.vstack(x['Drugs'] for x in shards)
    Labs = scipy.sparse.vstack(x['Labs'] for x in shards)
    Linked_Side_Effects = scipy.sparse.vstack(x['Linked_Side_Effects'] for x in shards)
    PATID = np.array(flatten(x['patid'] for x in shards))
    DS = np.array(flatten(x['ds'] for x in shards))

    log_step('Stacking shards', t_stacking)
    
    print('Adding PATID + DS') 
    t_adding = time()
    # ORDER:
    new = add_in_parallel(PATID,DS, n_jobs=n_jobs, n_chunks=n_chunks)
    #new2 = np.core.defchararray.add(PATID, DS)
    
    log_step('Adding PATID + DS', t_adding)
   
    t_argsort = time() 
    print('Get new order')
    order = np.argsort(new, kind='stable')
    log_step('Argsorting', t_argsort)

    # reordering:
    print('Processing Proc, Diag, Dr, La, LSE') 
    t_reorder = time() 
    Proc = Procedures[order,:]
    log_step('Reordering Procedures', t_reorder)
    
    #check_reordering(Procedures, Proc, order, n_tests=10000)
    t_reorder = time() 
    process_in_parallel(Proc, 'procedures', n_jobs=n_jobs, n_chunks=n_chunks)
    log_step('Writing Procedures', t_reorder)

    t_reorder = time() 
    Diag = Diagnoses[order,:]
    log_step('Reordering Diagnoses', t_reorder)

    t_reorder = time() 
    #check_reordering(Diagnoses, Diag, order, n_tests=10000)
    process_in_parallel(Diag, 'diagnoses', n_jobs=n_jobs, n_chunks=n_chunks)
    log_step('Writing Diagnoses', t_reorder)

    t_reorder = time() 
    Dr = Drugs[order,:]
    log_step('Reordering Drugs', t_reorder)

    #check_reordering(Drugs, Dr, order, n_tests=10000)
    t_reorder = time() 
    process_in_parallel(Dr, 'drugs', n_jobs=n_jobs, n_chunks=n_chunks)
    log_step('Writing Drugs', t_reorder)

    t_reorder = time() 
    La = Labs[order,:]
    log_step('Reordering Labs', t_reorder)
    
    t_reorder = time() 
    #check_reordering(Labs, La, order, n_tests=10000)
    process_in_parallel(La, 'labs', n_jobs=n_jobs, n_chunks=n_chunks)
    log_step('Writing Labs', t_reorder)
    
    t_reorder = time() 
    LSE = Linked_Side_Effects[order,:]
    log_step('Reordering LSE', t_reorder)
   
    t_reorder = time() 
    #check_reordering(Linked_Side_Effects, LSE, order, n_tests=10000)
    process_in_parallel(LSE, 'linked_side_effects', n_jobs=n_jobs, n_chunks=n_chunks)
    log_step('Writing LSE', t_reorder)

    print('processsing pid, ds')
    t_reorder = time() 
    PATID_r = PATID[order]
    log_step('Reordering patids', t_reorder)

    #check_reordering_array(PATID, PATID_r, order, n_tests=500)
    t_reorder = time() 
    DS_r = DS[order]
    log_step('Reordering ds', t_reorder)

    #check_reordering_array(DS, DS_r, order, n_tests=500) 
    t_reorder = time()
    process_array_in_parallel(DS_r, 'datestamps', n_jobs=n_jobs, n_chunks=n_chunks)
    process_array_in_parallel(PATID_r, 'patids', n_jobs=n_jobs, n_chunks=n_chunks)
    # np.savez( '../../data/reordered/indices_r.npz', PATID=PATID_r, DS=DS_r)
    log_step('Writing patids & ds', t_reorder)

    print('FINISHED')
 
if __name__ == "__main__":
    fire.Fire(main)
