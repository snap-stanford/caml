import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse

#custom
from .xlearner import MLPLateConcatModelLayernorm


class CaMLModel(MLPLateConcatModel):
    """ 
    CaML Model class. 
    """
    def __init__(*args, **kwargs):
        super(CaMLModel, self).__init__(*args, **kwargs)

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser])
        parser.add_argument('--caml_k', type=int, default=50) 
        return parser

