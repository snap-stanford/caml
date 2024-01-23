import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from causal_singletask.consts_and_utils import (DATA_LOCATION, MODEL_OUTPUTS, VOCABULARY, collateModels)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import hashlib
import pickle
import numpy as np

class XTrtLearner:
    def __init__(
        self,
        model_path: str = MODEL_OUTPUTS + 'x_learners/',
        m0 = None,
        m1 = None,
        mx1 = None
    ):
        os.makedirs(model_path,exist_ok=True)
        self.m0 = m0
        self.mx1 = mx1

        if self.m0 == None:
            self.m0 = RandomForestClassifier(random_state=0, n_estimators=150, max_depth=30,n_jobs=75,verbose=True)

        if self.mx1 == None:
            self.mx1 = RandomForestRegressor(random_state=0, n_estimators=150, max_depth=30,n_jobs=75,verbose=True)
        self.model_path = model_path
        self.model_hash = hashlib.md5(pickle.dumps(self.m0)).hexdigest()+hashlib.md5(pickle.dumps(self.mx1)).hexdigest()

    def train(self, train_subset):

        data_hash = hashlib.md5(pickle.dumps(train_subset)).hexdigest()

        path = self.model_path + data_hash+self.model_hash+'.pkl'

        if not os.path.isfile(path):
            print ("Writing X-trt model to path {}".format(path))
            train_treatment_mask = train_subset['W'].astype(bool)
            X_train = train_subset['X']
            y_train = train_subset['y']
            print('fitting m0')
            self.m0.fit(X_train[~train_treatment_mask,:], y_train[~train_treatment_mask])
            X_train_1 = X_train[ train_subset['W'].nonzero()[0],:]
            tau_train_1 = y_train[ train_subset['W'].nonzero()[0]].astype(np.float32) - self.m0.predict_proba(X_train_1)[:, 1]
            print('fitting mx1')
            self.mx1.fit(X_train_1, tau_train_1)

            with open(path, 'wb') as handle:
                pickle.dump(self.mx1,handle)


        else:
            print ("Loaded X-trt model from path {}".format(path))
            with open(path, 'rb') as handle:
                self.mx1 = pickle.load(handle)


    def predict(self, X):
        return self.mx1.predict(X)


class MLPLayer(nn.Module):
    """ Custom MLP Layer
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.,
        batch_norm: bool = False,
        layer_norm: bool = False,
        residual: bool = False,
        activation_fn: nn.Module = nn.ReLU,
    ):
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        assert layer_norm + batch_norm <= 1
        self.residual = residual
        super(MLPLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            activation_fn(),
            nn.Dropout(dropout)
        )
        if batch_norm:
            self.bn = nn.BatchNorm1d(num_features=output_dim)
        elif layer_norm:
            self.bn = nn.LayerNorm(output_dim)
        if residual:
            assert input_dim == output_dim
        #self.layer = nn.Sequential(*steps)

    def forward(self, x):
        if self.residual:
            x = x + self.layer(x)
        else:
            x = self.layer(x)
        if self.batch_norm:
            # bn requires (N,C) or (N,C,L) input, so we flatten empty dim at position 1 for it:
            x_shape = x.shape # keep track of old shape
            x = x.view([x.shape[0], x.shape[-1]])
            x = self.bn(x)
            x = x.view(x_shape) # get again (N,1,C) format
        elif self.layer_norm:
            x = self.bn(x)
        return x

class MLPModel(nn.Module):
    """ MLP model backbone for the X-TrEAtment (EXTRA) learner
    """
    def __init__(
            self,
            input_dim,
            output_dim,
            dropout = 0.2,
            model_dim = 256,
            n_layers = 2,
            batch_norm = False,
            layer_norm = False,
            regress = False,
            residual = False,
            layer_norm_inner_layers = False,
    ):
        super(MLPModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.model_dim = model_dim
        self.n_layers = n_layers
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.regress = regress
        self.residual = residual
        assert not (batch_norm and layer_norm_inner_layers)

        self.encoder = MLPLayer(input_dim, model_dim, 
                dropout, batch_norm, layer_norm=layer_norm
        )
        self.layers = nn.ModuleList(
            [ MLPLayer(model_dim, model_dim, dropout, #batch_norm=batch_norm, 
                layer_norm=layer_norm_inner_layers, residual=residual) for _ in range(n_layers)]
        )#  batch_norm=True
        final_activation_fn = nn.Tanh if regress else nn.Sigmoid
        print(f'Using {final_activation_fn}')
        self.final = nn.Sequential(
            nn.Linear(model_dim, output_dim),
            final_activation_fn()
        )

    def forward(self, x):
        # Encode high-dim features in low-dim hidden repr.
        x = self.encoder(x)
        # Feed encoded repr. through MLP layers
        for layer in self.layers:
            x = layer(x)
        # Apply final layer with activation depending on regression task (or not)
        return self.final(x)

class MLPLateConcatModel(nn.Module):
    """ MLP model backbone for the X-TrEAtment (EXTRA) learner.
        This model integrates task embeddings later in the network, such
        that the task embeddings are not diluted among the high-dim input features
    """
    def __init__(
            self,
            data_input_dim,
            task_input_dim,
            output_dim,
            dropout = 0.0,
            model_dim = 256,
            n_layers = 2,
            batch_norm = False,
            regress = False,
            residual = False
    ):
        super(MLPLateConcatModel, self).__init__()
        # data and task embeddings are encoded separately
        self.encoder = MLPLayer(data_input_dim, model_dim, dropout, batch_norm)
        # encoder = data_encoder, but we keep naming for consistent layer name in l1 loss
        self.task_encoder = MLPLayer(task_input_dim, model_dim//2, dropout, batch_norm)
        self.layers = nn.ModuleList(
            [ MLPLayer(model_dim + model_dim//2 if i==0 else model_dim,  # first layer needs to catch double dim
                model_dim, dropout,
                residual=residual if i > 0 else False) for i in range(n_layers)]
        )#  batch_norm=True
        final_activation_fn = nn.Tanh if regress else nn.Sigmoid
        print(f'Using {final_activation_fn}')
        self.final = nn.Sequential(
            nn.Linear(model_dim, output_dim),
            final_activation_fn()
        )

    def forward(self, x, t):
        """ Inputs:
            -x: data embedding [batch, <empty>, features]
            -t: task embedding [batch, <empty>, features]
        """
        # Encode high-dim features in low-dim hidden repr.
        x = self.encoder(x)
        t = self.task_encoder(t)

        # Concatenate encoded data and task embeddings at innermost (feature) dim:
        x = torch.cat([x,t], -1)
        # Feed encoded repr. through MLP layers
        for layer in self.layers:
            x = layer(x)
        # Apply final layer with activation depending on regression task (or not)
        return self.final(x)


class MLPModelNoFinalAct(MLPModel):
    """
    Subclass of MLPModel w/o final activation, instead
    raw logit output.
    assumes parameters are passed with names in signatures
    (e.g. input_dim=10, output_dim=3, model_dim= 256,..)
    """
    def __init__(self, *args, **kwargs):
        super(MLPModelNoFinalAct, self).__init__(*args, **kwargs)
        self.final = nn.Linear(self.model_dim, self.output_dim)

class MLPParamPredModel(MLPModel):
    """
    Subclass of MLPModel w/ final layer specific to param preed.
    assumes parameters are passed with names in signatures
    (e.g. input_dim=10, output_dim=3, model_dim= 256,..)
    """
    def __init__(self, final_norm='layer_norm', *args, **kwargs):
        super(MLPModelNoFinalAct, self).__init__(*args, **kwargs)
        if final_norm == 'layer_norm':
            final_step = nn.LayerNorm(self.output_dim)
        elif final_norm == 'tanh':
            final_step = nn.Tanh()
        else:
            raise ValueError()

        self.final = nn.Sequential(
            nn.Linear(self.model_dim, self.output_dim),
            final_step
        )



class MLPLateConcatModelLayernorm(nn.Module):
    """ MLP model backbone for the X-TrEAtment (EXTRA) learner.
        This model integrates task embeddings later in the network, such
        that the task embeddings are not diluted among the high-dim input features
    """
    def __init__(
            self,
            data_input_dim,
            task_input_dim,
            output_dim,
            dropout = 0.0,
            model_dim = 256,
            n_layers = 2,
            regress = False,
            residual = False,
            final_activation = True,
            layer_norm_inner_layers = False,
    ):
        super(MLPLateConcatModelLayernorm, self).__init__()
        # data and task embeddings are encoded separately
        self.encoder = MLPLayer(data_input_dim, model_dim, dropout, layer_norm=True)
        # encoder = data_encoder, but we keep naming for consistent layer name in l1 loss
        self.task_encoder = MLPLayer(task_input_dim, model_dim//2, dropout, layer_norm=True)
        self.layers = nn.ModuleList(
            [ MLPLayer(model_dim + model_dim//2 if i==0 else model_dim,  # first layer needs to catch double dim
                model_dim, dropout, layer_norm=layer_norm_inner_layers,
                residual=residual if i > 0 else False) for i in range(n_layers)]
        )#  batch_norm=True
        final_activation_fn = nn.Tanh if regress else nn.Sigmoid
        if final_activation:
            print(f'Using {final_activation_fn}')
            self.final = nn.Sequential(
                nn.Linear(model_dim, output_dim),
                final_activation_fn()
            )
        else:
            self.final = nn.Linear(model_dim, output_dim) 

    def forward(self, x, t):
        """ Inputs:
            -x: data embedding [batch, <empty>, features]
            -t: task embedding [batch, <empty>, features]
        """
        # Encode high-dim features in low-dim hidden repr.
        x = self.encoder(x)
        t = self.task_encoder(t)

        # Concatenate encoded data and task embeddings at innermost (feature) dim:
        x = torch.cat([x,t], -1)
        # Feed encoded repr. through MLP layers
        for layer in self.layers:
            x = layer(x)
        # Apply final layer with activation depending on regression task (or not)
        return self.final(x)
