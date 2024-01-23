import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from .xlearner import MLPModel
import numpy as np 

class HyperNet(nn.Module):
    """
    Hypernetwork implementation 
    """
    def __init__(self, 
            mnet: object = None,
            hnet_model_cls: str ='MLPModel',
            hnet_parameters: dict = {},
    ):
        super().__init__()
        assert hnet_model_cls == 'MLPModel' #TODO: makes this general with getattr()
        assert mnet is not None
        assert 'input_dim' in hnet_parameters.keys()
        output_dim = self.shapes_to_num_weights(
                mnet.parameters()
        )
        hnet_parameters.update({'output_dim': output_dim})
        self.model = MLPModel(**hnet_parameters)

    def forward(self, X):
        return self.model(X) 

    def shapes_to_num_weights(self, params):
        """ Takes list or generator of parameters and returns 
            # of parameters (int).
        """
        return int(np.sum([np.prod(l.shape) for l in params]))

class MainNet(nn.Module):
    """
    Main network (of a hypernetwork)
    """
    def __init__(self,
        mnet_model_cls: str = 'MLPModel',
        mnet_parameters: dict = {}
    ):
        assert mnet_model_cls == 'MLPModel' #TODO: makes this general with getattr()
        self.model = MLPModel(**mnet_parameters)
 
    def forward(self, X, weights=None):
        if weights is None:
            self._warn_no_weights()

        self.load_weights(weights)

    def _load_weights(self, weights: torch.tensor):
        pass

    def _warn_no_weights(self):
        raise Warning(
            """ No weights provided to the main 
                network, using default (initialized) parameters.
            """
        )

