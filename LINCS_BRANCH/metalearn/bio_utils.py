import pandas as pd
import numpy as np
import os
import pickle as pkl
from torch.utils.data import Dataset
import torch
from typing import Union
from scipy.sparse import csr_matrix, coo_matrix

# Try, except because some tasks might not have a train/val set (because of filtering of cell names)
def mean_batch(losses):
    try:
        return mean(losses)
    except:
        return np.NaN





def generate_trainvaltest_metadata_split(tasks_folder_path="BIO_preprocessing/BIO_datasets/tasks/",
                                         experiment_name="debug",
                                         seed=42,
                                         train_test_val_split=[0.95, 0.025, 0.025]):
    df_tasks = pd.read_csv(f"{tasks_folder_path}DrugBankIDs.csv")
    train_test_val_split = np.multiply(
        train_test_val_split, len(df_tasks)).astype(int)
    df_tasks = df_tasks.sample(frac=1, random_state=seed)  # shuffle
    df_tasks_train = df_tasks[:train_test_val_split[0]]
    df_tasks_val = df_tasks[train_test_val_split[0]:(
            train_test_val_split[0] + train_test_val_split[1])]
    df_tasks_test = df_tasks[(
                                     train_test_val_split[0] + train_test_val_split[1]):]
    print(
        f"Split meta-dataset into {len(df_tasks_train)} train tasks, {len(df_tasks_val)} val task, {len(df_tasks_test)} test tasks")

    # Save train val split of drugs in a separate file
    # experiment_path = f"{tasks_folder_path}experiment_splits/{experiment_name}"
    # try:
    #     os.mkdir(experiment_path)
    # except:
    #     pass
    # df_tasks_train.to_csv(f"{experiment_path}/df_tasks_train.csv", index=False)
    # df_tasks_val.to_csv(f"{experiment_path}/df_tasks_val.csv", index=False)
    # df_tasks_test.to_csv(f"{experiment_path}/df_tasks_test.csv", index=False)
    return df_tasks_train,df_tasks_val,df_tasks_test

def get_data_for_task(drug,tasks_folder_path,cell_names_train,cell_names_val,cell_names_test):
    tau = pd.read_csv(f"{tasks_folder_path}{drug}_tau.csv")
    X = pd.read_csv(f"{tasks_folder_path}{drug}_X.csv")
    with open(f"{tasks_folder_path}X_order_columns.pkl", 'rb') as f:
        X_ordered_columns = pkl.load(f)
    with open(f"{tasks_folder_path}tau_order_columns.pkl", 'rb') as f:
        tau_ordered_columns = pkl.load(f)
    assert (tau.columns.to_list() == tau_ordered_columns)
    assert (X.columns.to_list() == X_ordered_columns)
    X = X[X_ordered_columns]
    tau = tau[tau_ordered_columns]
    assert (tau.columns.to_list() == tau_ordered_columns)
    assert (X.columns.to_list() == X_ordered_columns)

    def filter_out_cellnames_and_preprocess(X, tau, cell_names):
        X_filter = X[X["cell_iname"].isin(cell_names)]
        tau_filter = tau[X["cell_iname"].isin(cell_names)]
        X_filter = X_filter.drop(columns=["cell_iname", "DepMap_ID"])
        X_filter = np.array(X_filter.values)
        tau_filter = np.array(tau_filter.values)
        if X_filter.shape[0] == 0:
            inner_bool = False
        else:
            inner_bool = True
        return X_filter, tau_filter, inner_bool

    X_train, tau_train, inner_train_bool = filter_out_cellnames_and_preprocess(X, tau, cell_names_train)
    X_val, tau_val, inner_val_bool = filter_out_cellnames_and_preprocess(X, tau, cell_names_val)
    X_test, tau_test, inner_test_bool = filter_out_cellnames_and_preprocess(X, tau, cell_names_test)
    #X_train = X_train.drop(columns=["cell_iname", "DepMap_ID"])
    #X_val = X_val.drop(columns=["cell_iname", "DepMap_ID"])
    #X_test = X_test.drop(columns=["cell_iname", "DepMap_ID"])
    # X = torch.tensor(X.values)
    # tau = torch.tensor(tau.values)
    #X_train = np.array(X_train.values)
    #X_val = np.array(X_val.values)
    #X_test = np.array(X_test.values)
    #tau = np.array(tau.values)
    #Baselines
    #baselines = []
    #add baseline of always predicting 0
    #zero_baseline = np.zeros(tau.shape)
    #baselines.append(zero_baseline)
    return X_train, tau_train, X_val, tau_val, X_test, tau_test, inner_train_bool, inner_val_bool, inner_test_bool 

def get_index_for_most_diff_genes(drug,tasks_folder_path):
    with open(f"{tasks_folder_path}{drug}_20most_diff_genes_index.pkl", 'rb') as f:
        n20_most_diff_genes_index = pkl.load(f)
    with open(f"{tasks_folder_path}{drug}_50most_diff_genes_index.pkl", 'rb') as f:
        n50_most_diff_genes_index = pkl.load(f)
    return n20_most_diff_genes_index, n50_most_diff_genes_index

class TaskEmbeddingMapper:
    """
    Maps DrugBankID to task embedding
    """

    def __init__(self, ids_path:str, embeddings_path:str, random:bool):
        self.ids_path = ids_path
        self.embeddings_path = embeddings_path
        self.ids = self.read_pickle(ids_path)
        self.embeddings = self.read_pickle(embeddings_path)
        self.random = random

        #self.

    def read_pickle(self, path):
        with open(path, 'rb') as f:
            return pkl.load(f)

    def __call__(self, DrugBankID):
            if DrugBankID == 'NULL':
                task_embedding =  self.embeddings[0]
                task_embedding = np.zeros_like(task_embedding)
                return task_embedding

            index = self.ids.index([DrugBankID])
            task_embedding =  self.embeddings[index]

            if self.random:
                # create random task embedding in the same shape:
                seed = int(str(hash(DrugBankID))[-9:]) # use last 9 digits of hash
                rs = np.random.RandomState(seed)
                task_embedding = rs.randn(task_embedding.shape[0])
                print(f'Using random task embedding of shape {task_embedding.shape}')
            return task_embedding

class BioSINTaskDataset(Dataset):
    """
    Custom Dataset class for scipy sparse matrix that adds task embedding to input features
    """
    def __init__(
        self, data:Union[np.ndarray, coo_matrix, csr_matrix, np.ndarray,torch.Tensor],
        task_embedding: np.ndarray,
        *tensors: np.ndarray,
    ):

        assert all(data.shape[0] == tensor.shape[0] for tensor in tensors), "Size mismatch between tensors"

        self.data = torch.tensor(data,dtype=torch.float32)
        self.task_embedding = torch.tensor(task_embedding,dtype=torch.float32)
        self.tensors = [torch.tensor(x, dtype=torch.float32) for x in tensors]


    def __getitem__(self, index:int):
        data = self.data[index]
        task_embedding = self.task_embedding[index]
 
        return (data, task_embedding, *[tensor[index] for tensor in self.tensors])

    def __len__(self):
        return self.data.shape[0]



class BioSLearnerDataset(Dataset):
    """
    Custom Dataset class for scipy sparse matrix
    """
    def __init__(self, data:Union[np.ndarray, coo_matrix, csr_matrix], 
                       data2:Union[np.ndarray, coo_matrix, csr_matrix], 
                 *tensors: np.ndarray, 
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



class BioTaskDataset(Dataset):
    """
    Custom Dataset class for BIO dataset
    """

    def __init__(
            self, data: Union[np.ndarray, coo_matrix, csr_matrix, torch.Tensor],
            task_embedding: Union[np.ndarray, torch.Tensor],
            concatenate: bool,
            taus: Union[np.ndarray, torch.Tensor],
    ):
        assert data.shape[0] == taus.shape[0]
        # self.data = self.data.astype(np.float32)
        self.data = data
        # self.task_embedding = task_embedding.astype(np.float32)
        self.task_embedding = task_embedding

        if concatenate:
            self.data = np.hstack([self.data, self.task_embedding[[0] * self.data.shape[0], :]])

        self.concatenate = concatenate
        # self.taus = taus.astype(np.float32)
        self.taus = taus

        # Convert to correct type
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.taus = torch.tensor(self.taus, dtype=torch.float32)
        self.task_embedding = torch.tensor(self.task_embedding, dtype=torch.float32)
        self.task_embedding = torch.reshape(self.task_embedding, (-1,))

    def __getitem__(self, index: int):
        if self.concatenate:
            # task embedding is in the data
            return (self.data[index], self.taus[index])
        else:
            return (self.data[index], self.task_embedding, self.taus[index])

    def __len__(self):
        return self.data.shape[0]


#TODO: adjust for bio
class BioDataset(Dataset):
    """
    Custom Dataset class for scipy sparse matrix
    """
    def __init__(
            self, data: Union[np.ndarray, coo_matrix, csr_matrix, torch.Tensor],
            taus: Union[np.ndarray, torch.Tensor],
    ):

        assert data.shape[0] == taus.shape[0]
        self.data = data
        self.taus = taus

        # Convert to correct type
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.taus = torch.tensor(self.taus, dtype=torch.float32)

    def __getitem__(self, index: int):
        return (self.data[index], self.taus[index])

    def __len__(self):
        return self.data.shape[0]