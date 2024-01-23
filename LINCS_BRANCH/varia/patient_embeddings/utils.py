"""Some utility functions for pytorch."""
import math
import json
import numpy as np
import torch
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import TensorBoardLogger
import pandas as pd
from pathlib import Path

def variable_length_collate(batch):
    """Combine multiple instances of irregular lengths into a batch."""
    # This converts the batch from [{'a': 1, 'b': 2}, ..] format to
    # {'a': [1], 'b': [2]}
    transposed_items = {
        key: list(map(lambda a: a[key], batch))
        for key in batch[0].keys()
    }

    transposed_items['id'] = np.array(transposed_items['id'], dtype=np.int64)
    transposed_items['ds'] = np.array(transposed_items['ds'], dtype=np.int64)
    transposed_items['id_to_index'] = np.array(transposed_items['id_to_index'], dtype=np.int64)

    # TODO: check that lengths are correct!
    lengths = np.array(list(map(len, transposed_items['times']))) 
    transposed_items['lengths'] = lengths
    max_len = max(lengths)

    # Pad labels with -100 as this is the mask value for nll_loss

    for key in ['data', 'times']: # TODO: add times_embedded?
        if key not in transposed_items.keys():
            continue
        dims = len(transposed_items[key][0].shape)
        if dims == 1:
            padded_instances = [
                np.pad(instance, ((0, max_len - len(instance)),),
                       mode='constant')
                for instance in transposed_items[key]
            ]
        elif dims == 2:
            padded_instances = [
                np.pad(instance, ((0, max_len - len(instance)), (0, 0)),
                       mode='constant')
                for instance in transposed_items[key]
            ]
        else:
            raise ValueError(
                f'Unexpected dimensionality of instance data: {dims}')

        transposed_items[key] = np.stack(padded_instances, axis=0) 

    transposed_items['times'] = transposed_items['times'].astype(np.int16)
    transposed_items['data'] = transposed_items['data'].astype(np.float32)
    transposed_items['lengths'] = transposed_items['lengths'].astype(np.int16)
    # TODO: decide whether labels should be accounted for here
    #transposed_items['labels'] = transposed_items['labels'].astype(np.float32)
 
    tensor_dict = {
        key: torch.from_numpy(data) for key, data in
        transposed_items.items()
    }
    # tcn would require: [batch, channels, length]
    # tensor_dict['data'] = tensor_dict['data'].permute([0,2,1])
    return tensor_dict
