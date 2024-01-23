import numpy as np
from torch.utils.data import Dataset
import pickle
from sklearn.model_selection import train_test_split
import os

class MPDatasetInMemory(Dataset):
    """ Dataset for first drug side effect pair (MTX & PC)"""
    def __init__(self, path='../data/full/data.pkl',
            split=None):
        self.path = path 
        self.d = self._load_data(path) # entire dataset
        self.ids = list(self.d['id_to_index'].keys()) 
        print('Split not implemented yet!')

    def __getitem__(self, index):
        current_id = self.ids[index]
        start, end = self.d['id_to_index'][current_id]
        return {
            'data': self.d['data'][start:end,:].todense(), # data in dense format
            'ds':   self.d['ds'][start],  # start date stamp
            'times': self.d['times'][start:end], # times (relative, in days)
            'id':   np.array(
                        self.d['ids'][start]
                    ), # patient id
            'id_to_index': np.array( [start, end]) 
            # indices pointing to full dataset 
        }
    def __len__(self):
        return len(self.ids) # on the patient level

    def _load_data(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def get_split(self, random_state=None):
        print(f'Creating a split..')
        train_indices, test_indices = train_test_split(
            range(len(self.ids)),
            train_size=0.999,
            random_state=random_state
        )
        return train_indices, test_indices

class MPDemoDatasetInMemory(MPDatasetInMemory):
    def __init__(self, path='../data/demo/data.pkl', split=None, **kwargs):
        super().__init__(path, split, **kwargs)


class ShardedDataset(Dataset):
    """ Dataset which is formatted in shards (to save memory)"""
    def __init__(self, path='../data/demo',
            split=None):
        self.path = path 
        self.id_to_shard = self._load_mapping(path) # entire dataset
        self.ids = list(self.id_to_shard.keys()) 
        print('Split not implemented yet!')

    def __getitem__(self, index):
        current_id = self.ids[index]
        # get shard of current id
        shard_id = self.id_to_shard[current_id]
        # load shard:
        d = self._load_data(
            os.path.join(
                self.path, f'shard_{shard_id}.pkl' 
            )
        )
        start, end = d['id_to_index'][current_id]
        return {
            'data': d['data'][start:end,:].todense(), # data in dense format
            'ds':   np.array(d['ds'][start]),  # start date stamp
            'times': np.array(d['times'][start:end]), # times (relative, in days)
            'id':   np.array(
                        d['ids']
                    )[start], # patient id
            'id_to_index': np.array( [start, end]) 
            # indices pointing to full dataset 
        }
    def __len__(self):
        return len(self.ids) # on the patient level

    def _load_mapping(self, path):
        with open(
            os.path.join(
                path, 'id_mapping.pkl'
            ), 'rb') as f:
            return pickle.load(f)

    def _load_data(self, full_path):
        with open(full_path, 'rb') as f:
            return pickle.load(f)

    def get_split(self, random_state=None):
        print(f'Creating a split..')
        train_indices, test_indices = train_test_split(
            range(len(self.ids)),
            train_size=0.999,
            random_state=random_state
        )
        return train_indices, test_indices

class MPDataset(ShardedDataset):
    def __init__(self, path='../data/full_5000', split=None, **kwargs):
        super().__init__(path, split, **kwargs)

class MPDemoDataset(ShardedDataset):
    def __init__(self, path='../data/demo', split=None, **kwargs):
        super().__init__(path, split, **kwargs)






if __name__ == "__main__":
    pass

