""" Utility modules for task definition, loading, sampling"""
import numpy as np
import pandas as pd
import os
import pickle
import torch
from torch.utils.data import Dataset
from typing import Union
from scipy.sparse import (random,
                          coo_matrix,
                          csr_matrix,
                          vstack)
from scipy import sparse
from torch.utils.data.dataloader import default_collate
from pdb import set_trace as bp

task_embedding_paths = {
    'distmult': 'task_embedding/tasks/task_embeddings_dict_distmult.pkl',
    'stargraph': 'task_embedding/tasks/task_embeddings_dict_stargraph256.pkl'
}

class TaskSampler:
    """ class to sample individual task ids """

    def __init__(self, tasks: list, seed=42):
        """ init with list of tasks
            e.g. ['task_0_0','task_0_1',..]
        """
        self.tasks = tasks
        np.random.seed(seed)

    def __iter__(self):
        return self

    def __next__(self):
        return np.random.choice(self.tasks)


class TaskLoader:
    """ Loads task information (tasks, task embeddings) for a given task"""

    def __init__(
        self,
        task_path: str,
        task_embedding_path: str,
        drug_path: str,
    ):
        self.tasks = self._load_pickle(task_path)
        self.task_embeddings = self._load_pickle(task_embedding_path)
        self.drugs = pd.read_csv(drug_path)

    def _load_pickle(self, path: str):
        with open(path,'rb') as f:
            return pickle.load(f)

    def __call__(self, task_id: str):
        """ task_id: e.g. 'task_0_2' (side effect 0, drug 2)"""
        d = {}
        drugs, tasks= self.drugs, self.tasks
        d['drugbank_id'] = tasks[task_id]['drugbank_id']
        d['drug_name'] = drugs[drugs['index'] == tasks[task_id]['drugs']]['name'].values[0]
        d['side_effect_id'] = tasks[task_id]['side_effects'] #currently just one
        d['task_embedding'] = self.task_embeddings[task_id]
        return d

class DrugEmbeddingMapper(TaskLoader):
    def __init__(self, *args, **kwargs):
        super(DrugEmbeddingMapper, self).__init__(*args, **kwargs)
        self.drug_to_task = {d['drugbank_id']: task for task, d in self.tasks.items()}

    def __call__(self, drug_id, verbose=False):
        task_id = self.drug_to_task[drug_id]
        if not verbose:
            return self.task_embeddings[task_id]
        else:
            return super(DrugEmbeddingMapper, self).__call__(task_id)


class TaskEmbeddingMapper:
    """
    Mapping from set of drug_bank ids to task embedding,
    handling aggregation of drug combinations on the fly!
    """
    def __init__(
            self,
            embedding_path: str,
            aggregation: str = 'sum', # mean
            random: bool = False
        ):
        """
        TODO: add docu
        - random: return random embedding, determined by task id
        - random: return random embedding, determined by task id 
        """
        self.aggregation = aggregation
        self.random = random
        if random:
            print(f'Ignoring task embeddings and returning random ones.')
        print(f'Loading task embeddings from: {embedding_path}')
        self.d = self._read_pickle(embedding_path)


        # Create a drug embedding that is the mean of all drugs
        sumAll = None
        for drug_id in self.d.keys():
            if type(sumAll) == type(None):
                sumAll = self.d[drug_id]
            else:
                sumAll += self.d[drug_id]
        sumAll /= len(self.d.keys())
        self.null_drug_embedding = sumAll

        assert aggregation in ['mean', 'sum']

    def _read_pickle(self, fname):
        with open(fname, 'rb') as f:
            return pickle.load(f)

    def _input_validation(self, drug_ids: str):
        """
        Check that input string is sorted properly
        """
        combo_list = drug_ids.split('_') # 'DB00316_DB00956' --> ['DB00316','DB00956']
        assert sorted(combo_list) == combo_list

    def _get_drug_embedding(self, drug_id):
        if drug_id not in self.d.keys() and drug_id!='NULL':
            raise ValueError(f'Drug {drug_id} not in dictionary of tasks!')

        if drug_id == 'NULL':
            # from IPython.core.debugger import Pdb; Pdb().set_trace()
            return self.null_drug_embedding

        return self.d[drug_id]

    def __call__(self, drug_ids: str):
        """
        We expect a drug combination as a string of drugbank ids,
            e.g. 'DB00316_DB00956'
        """
        self._input_validation(drug_ids)

        combo_list = drug_ids.split('_')
        embeddings = [
            self._get_drug_embedding(drug) for drug in combo_list
        ]
        agg_func = getattr(np, self.aggregation)
        task_embedding = agg_func(embeddings, axis=0)
        if self.random:
            # create random task embedding in the same shape:
            seed = int(str(hash(drug_ids))[-9:]) # use last 9 digits of hash
            rs = np.random.RandomState(seed)
            task_embedding = rs.randn(task_embedding.shape[0])
            print(f'Using random task embedding of shape {task_embedding.shape}')
        return task_embedding


class SINTaskDataset(Dataset):
    """
    Custom Dataset class for scipy sparse matrix that adds task embedding to input features
    """
    def __init__(
        self, data:Union[np.ndarray, coo_matrix, csr_matrix],
        task_embedding: np.ndarray,
        *tensors: torch.Tensor,
    ):

        assert all(data.shape[0] == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"

        # Transform data coo_matrix to csr_matrix for indexing
        if type(data) == coo_matrix:
            self.data = data.tocsr()
        else:
            self.data = data

        self.task_embedding = task_embedding
        self.tensors = tensors


    def __getitem__(self, index:int):
        data = self.data[index]
        task_embedding = self.task_embedding[index]
 
        return (data, task_embedding, *[tensor[index] for tensor in self.tensors])

    def __len__(self):
        return self.data.shape[0]




class ClaimsTaskDataset(Dataset):
    """
    Custom Dataset class for scipy sparse matrix that adds task embedding to input features
    """
    def __init__(
        self, data:Union[np.ndarray, coo_matrix, csr_matrix],
        task_embedding: np.ndarray,
        concatenate: bool,
        *tensors: torch.Tensor,
    ):

        assert all(data.shape[0] == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"

        # Transform data coo_matrix to csr_matrix for indexing
        if type(data) == coo_matrix:
            self.data = data.tocsr()
        else:
            self.data = data

        if concatenate:
            self.task_embedding = sparse.csr_matrix(task_embedding)
            self.data = sparse.hstack([self.data, self.task_embedding[[0]*self.data.shape[0],:]])
        else:
            self.task_embedding = task_embedding

        self.concatenate = concatenate
        self.tensors = tensors


    def __getitem__(self, index:int):
        data = self.data[index]
        if self.concatenate:
            # add task embedding:
            # data = sparse.hstack([data, self.task_embedding])
            return (data, *[tensor[index] for tensor in self.tensors])
        else:
            return (data, self.task_embedding, *[tensor[index] for tensor in self.tensors])

    def __len__(self):
        return self.data.shape[0]

def sparse_csr_to_tensor(data_batch):
    data_batch = sparse.vstack(data_batch).tocoo()
    data_batch = sparse_coo_to_tensor(data_batch)
    return data_batch


def sparse_coo_to_tensor(coo:coo_matrix):
    """
    Transform scipy coo matrix to pytorch sparse tensor
    """
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    shape = coo.shape

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    s = torch.Size(shape)

    return torch.sparse.FloatTensor(i, v, s)

def collateFun(col):
    if type(col[0]) == np.ndarray:
        return torch.FloatTensor(np.vstack(col))
    else:
        return default_collate(col)


def sparse_batch_collate(batch:list):
    """
    Collate function which to transform scipy coo matrix to pytorch sparse tensor
    """
    data_batch, *tensors = zip(*batch)

    if type(data_batch[0]) == csr_matrix:
        data_batch = sparse.vstack(data_batch).tocoo()
        data_batch = sparse_coo_to_tensor(data_batch)
    else:
        data_batch = torch.FloatTensor(data_batch)

    return data_batch, *[collateFun(tensor) for tensor in tensors]

def sparse_batch_collate2(batch:list):
    """
    Collate function which to transform scipy coo matrix to pytorch sparse tensor
    """
    data_batch, data_batch2, *tensors = zip(*batch)

    if type(data_batch[0]) == csr_matrix:
        data_batch = sparse.vstack(data_batch).tocoo()
        data_batch = sparse_coo_to_tensor(data_batch)
    else:
        data_batch = torch.FloatTensor(data_batch)

    if type(data_batch2[0]) == csr_matrix:
        data_batch2 = sparse.vstack(data_batch2).tocoo()
        data_batch2 = sparse_coo_to_tensor(data_batch2)
    else:
        data_batch2 = torch.FloatTensor(data_batch2)

    return data_batch, data_batch2, *[collateFun(tensor) for tensor in tensors]


def sparse_batch_collate3(batch:list):
    """
    Collate function which to transform scipy coo matrix to pytorch sparse tensor
    """
    data_batch, data_batch2, *tensors = zip(*batch)

    if type(data_batch[0]) == csr_matrix:
        data_batch = sparse.vstack(data_batch).tocoo()
        data_batch = sparse_coo_to_tensor(data_batch)
    else:
        data_batch = torch.FloatTensor(data_batch)

    if type(data_batch2[0]) == csr_matrix:
        data_batch2 = sparse.vstack(data_batch2).tocoo()
        data_batch2 = sparse_coo_to_tensor(data_batch2)
    else:
        data_batch2 = torch.FloatTensor(np.array(data_batch2))

    return data_batch, data_batch2, *[collateFun(tensor) for tensor in tensors]


def filter_for_used_drugs(df_tasks, used_drugs_path):
    " Filter step to ensure that we only use the 744 drugs that are available from the KG "
    used_drugs = pd.read_csv(used_drugs_path)
    index = df_tasks['Drugs'].apply(
        lambda str_of_drugs:
            all([d in list(used_drugs['id']) for d in str_of_drugs.split('_')
                ])
    )
    return df_tasks[index]

if __name__ == "__main__":

    # Example usage of TaskSampler / TaskLoader
    task_sampler = TaskSampler(['task_0_0', 'task_0_1', 'task_0_2'])

    add_path = lambda x: os.path.join('task_embedding/tasks/', x)

    task_loader = TaskLoader(
        task_path = add_path('tasks.pkl'),
        task_embedding_path = add_path('task_embeddings.pkl'),
        drug_path = add_path('used_drugs.csv'),
    )
    de_mapper = DrugEmbeddingMapper(
        task_path = add_path('tasks.pkl'),
        task_embedding_path = add_path('task_embeddings.pkl'),
        drug_path = add_path('used_drugs.csv'),
    )

    n = 3 # sampling n tasks

    print(f'Sampled tasks: ')
    for i, task_id in enumerate(task_sampler):
        print(f'Loading {task_id} ... ')
        print(task_loader(task_id))
        if i == n-1:
            break
