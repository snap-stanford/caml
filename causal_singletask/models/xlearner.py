import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from causal_singletask.consts_and_utils import (DATA_LOCATION, MODEL_OUTPUTS, VOCABULARY, collateModels)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import hashlib
import pickle
import numpy as np
from pdb import set_trace as bp
from sklearn.model_selection import cross_val_predict, KFold, train_test_split


####
####  Todo: Implement  X-learner, R-learner
####


class XLearner:
    def __init__(
        self,
        model_path: str = MODEL_OUTPUTS + 'x_learners_full/',
        m0 = None,
        m1 = None,
        g = None,
        mx0 = None,
        mx1 = None,
        args = None
    ):
        os.makedirs(model_path,exist_ok=True)
        self.m0 = m0
        self.m1 = m1
        self.mx1 = mx1
        self.mx0 = mx0
        self.g = g


        n_estimators = 150
        max_depth = 30
        min_samples_split = 2
        criterion_regress = 'squared_error'
        criterion_binary = 'gini'
        max_features = 'sqrt'

        if args != None:
            n_estimators = args.rf_n_estimators
            max_depth = args.rf_max_depth
            min_samples_split = args.rf_min_samples_split
            criterion_regress = args.rf_criterion_regress
            criterion_binary = args.rf_criterion_binary
            max_features = args.rf_max_features

        if self.m0 == None:
            self.m0 = RandomForestRegressor(random_state=0, n_estimators=n_estimators, max_depth=max_depth,n_jobs=75,verbose=True, min_samples_split=min_samples_split, criterion=criterion_regress, max_features=max_features)
        
        if self.m1 == None:
            self.m1 = RandomForestRegressor(random_state=0, n_estimators=n_estimators, max_depth=max_depth,n_jobs=75,verbose=True, min_samples_split=min_samples_split, criterion=criterion_regress, max_features=max_features)

        if self.mx0 == None:
            self.mx0 = RandomForestRegressor(random_state=0, n_estimators=n_estimators, max_depth=max_depth,n_jobs=75,verbose=True, min_samples_split=min_samples_split, criterion=criterion_regress, max_features=max_features)

        if self.mx1 == None:
            self.mx1 = RandomForestRegressor(random_state=0, n_estimators=n_estimators, max_depth=max_depth,n_jobs=75,verbose=True, min_samples_split=min_samples_split, criterion=criterion_regress, max_features=max_features)

        if self.g == None:
            self.g = RandomForestClassifier(random_state=0, n_estimators=n_estimators, max_depth=max_depth,n_jobs=75,verbose=True, min_samples_split=min_samples_split, criterion=criterion_binary, max_features=max_features)

        self.model_path = model_path
        self.model_hash = hashlib.md5(pickle.dumps(self.m0)).hexdigest()+hashlib.md5(pickle.dumps(self.m1)).hexdigest()+hashlib.md5(pickle.dumps(self.mx0)).hexdigest()+hashlib.md5(pickle.dumps(self.mx1)).hexdigest()+hashlib.md5(pickle.dumps(self.g)).hexdigest()



    def train(self, train_subset):

        data_hash = hashlib.md5(pickle.dumps(train_subset)).hexdigest()

        path = self.model_path + data_hash+self.model_hash+'.pkl'
        path1 = self.model_path + data_hash+self.model_hash+'_2.pkl'
        path2 = self.model_path + data_hash+self.model_hash+'_3.pkl'

        if not os.path.isfile(path):
            print ("Writing X-learner model to path {}".format(path))
            train_treatment_mask = train_subset['W'].astype(bool)
            X_train = train_subset['X']
            y_train = train_subset['y']

            print('fiting g')
            self.g.fit(X_train, train_subset['W']);

            print('fitting m0')
            self.m0.fit(X_train[~train_treatment_mask,:], y_train[~train_treatment_mask])
            print('fitting m1')
            self.m1.fit(X_train[train_treatment_mask,:], y_train[train_treatment_mask])

            d_train = np.where(train_subset['W']==0,  self.m1.predict(X_train) - y_train,   y_train - self.m0.predict(X_train))

            self.mx0.fit(X_train[~train_treatment_mask,:], d_train[train_subset['W']==0])
            self.mx1.fit(X_train[train_treatment_mask,:], d_train[train_subset['W']==1])

            with open(path, 'wb') as handle:
                pickle.dump(self.mx0,handle)
            with open(path1, 'wb') as handle:
                pickle.dump(self.mx1,handle)
            with open(path2, 'wb') as handle:
                pickle.dump(self.g,handle)


        else:
            print ("Loaded X-learner model from path {}".format(path))
            with open(path, 'rb') as handle:
                self.mx0 = pickle.load(handle)
            with open(path1, 'rb') as handle:
                self.mx1 = pickle.load(handle)
            with open(path2, 'rb') as handle:
                self.g = pickle.load(handle)

    def ps_predict(self, dataset, t): 
        return self.g.predict_proba(dataset)[:, t]

    def predict(self, X):
        return  (self.ps_predict(X,0)*self.mx0.predict(X) + self.ps_predict(X,1)*self.mx1.predict(X))


class XTrtLearner:
    def __init__(
        self,
        model_path: str = MODEL_OUTPUTS + 'x_learners/',
        m0 = None,
        m1 = None,
        mx1 = None,
        args = None
    ):
        os.makedirs(model_path,exist_ok=True)
        self.m0 = m0
        self.mx1 = mx1


        n_estimators = 150
        max_depth = 30
        min_samples_split = 2
        criterion_regress = 'squared_error'
        criterion_binary = 'gini'
        max_features = 'sqrt'

        if args != None:
            n_estimators = args.rf_n_estimators
            max_depth = args.rf_max_depth
            min_samples_split = args.rf_min_samples_split
            criterion_regress = args.rf_criterion_regress
            criterion_binary = args.rf_criterion_binary
            max_features = args.rf_max_features

        if self.m0 == None:
            self.m0 = RandomForestClassifier(random_state=0, n_estimators=n_estimators, max_depth=max_depth,n_jobs=75,verbose=True, min_samples_split=min_samples_split, criterion=criterion_binary, max_features=max_features)

        if self.mx1 == None:
            self.mx1 = RandomForestRegressor(random_state=0, n_estimators=n_estimators, max_depth=max_depth,n_jobs=75,verbose=True, min_samples_split=min_samples_split, criterion=criterion_regress, max_features=max_features)

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




class RLearner:
    def __init__(
        self,
        model_path: str = MODEL_OUTPUTS + 'r_learners/',
        g = None,
        outcome_learner = None,
        effect_learner = None,
        args = None,
        folds = 5
    ):
        os.makedirs(model_path,exist_ok=True)
        self.g = g
        self.outcome_learner = outcome_learner
        self.effect_learner = effect_learner
        self.folds = folds

        n_estimators = 150
        max_depth = 30
        min_samples_split = 2
        criterion_regress = 'squared_error'
        criterion_binary = 'gini'
        max_features = 'sqrt'

        if args != None:
            n_estimators = args.rf_n_estimators
            max_depth = args.rf_max_depth
            min_samples_split = args.rf_min_samples_split
            criterion_regress = args.rf_criterion_regress
            criterion_binary = args.rf_criterion_binary
            max_features = args.rf_max_features

        if self.g == None:
            self.g = RandomForestClassifier(random_state=0, n_estimators=n_estimators, max_depth=max_depth,n_jobs=75,verbose=True, min_samples_split=min_samples_split, criterion=criterion_binary, max_features=max_features)

        if self.outcome_learner == None:
            self.outcome_learner = RandomForestRegressor(random_state=0, n_estimators=n_estimators, max_depth=max_depth,n_jobs=75,verbose=True, min_samples_split=min_samples_split, criterion=criterion_regress, max_features=max_features)
        
        if self.effect_learner == None:
            self.effect_learner = RandomForestRegressor(random_state=0, n_estimators=n_estimators, max_depth=max_depth,n_jobs=75,verbose=True, min_samples_split=min_samples_split, criterion=criterion_regress, max_features=max_features)

        self.model_path = model_path
        self.model_hash = hashlib.md5(pickle.dumps(self.g)).hexdigest()+hashlib.md5(pickle.dumps(self.outcome_learner)).hexdigest()+hashlib.md5(pickle.dumps(self.effect_learner)).hexdigest()

    def train(self, train_subset):

        CV = KFold(n_splits=self.folds, shuffle=True, random_state=42)

        data_hash = hashlib.md5(pickle.dumps(train_subset)).hexdigest()

        path = self.model_path + data_hash+self.model_hash+'.pkl'

        if not os.path.isfile(path):
            print ("Writing R-learner model to path {}".format(path))
            train_treatment_mask = train_subset['W'].astype(bool)
            X_train = train_subset['X']
            y_train = train_subset['y']
            train_treatment =  train_subset['W']

            print('fitting yhat')

            yhat = cross_val_predict(self.outcome_learner, X_train, y_train, cv=CV, n_jobs=-1)
            print('fitting ehat')
            ehat = cross_val_predict(self.g, X_train, train_treatment, cv=CV, n_jobs=-1, method='predict_proba')[:, 1]
            labels = (y_train - yhat) / (train_treatment - ehat)
            weights = (train_treatment - ehat)**2 

            print('fitting effect learner to tau loss')
            self.effect_learner.fit(X_train, labels, sample_weight=weights)

            with open(path, 'wb') as handle:
                pickle.dump(self.effect_learner,handle)


        else:
            print ("Loaded R-learner model from path {}".format(path))
            with open(path, 'rb') as handle:
                self.effect_learner = pickle.load(handle)


    def predict(self, X):
        return self.effect_learner.predict(X)




class TLearner:
    def __init__(
        self,
        model_path: str = MODEL_OUTPUTS + 't_learners/',
        m0 = None,
        m1 = None,
        args = None
    ):
        os.makedirs(model_path,exist_ok=True)
        self.m0 = m0
        self.m1 = m1


        n_estimators = 150
        max_depth = 30
        min_samples_split = 2
        criterion_regress = 'squared_error'
        criterion_binary = 'gini'
        max_features = 'sqrt'

        if args != None:
            n_estimators = args.rf_n_estimators
            max_depth = args.rf_max_depth
            min_samples_split = args.rf_min_samples_split
            criterion_regress = args.rf_criterion_regress
            criterion_binary = args.rf_criterion_binary
            max_features = args.rf_max_features

        if self.m0 == None:
            self.m0 = RandomForestRegressor(random_state=0, n_estimators=n_estimators, max_depth=max_depth,n_jobs=75,verbose=True, min_samples_split=min_samples_split, criterion=criterion_regress, max_features=max_features)

        if self.m1 == None:
            self.m1 = RandomForestRegressor(random_state=0, n_estimators=n_estimators, max_depth=max_depth,n_jobs=75,verbose=True, min_samples_split=min_samples_split, criterion=criterion_regress, max_features=max_features)

        self.model_path = model_path
        self.model_hash = hashlib.md5(pickle.dumps(self.m0)).hexdigest()+hashlib.md5(pickle.dumps(self.m1)).hexdigest()

    def train(self, train_subset):

        data_hash = hashlib.md5(pickle.dumps(train_subset)).hexdigest()

        path = self.model_path + data_hash+self.model_hash+'.pkl'
        path2 = self.model_path + data_hash+self.model_hash+'_2.pkl'

        if not os.path.isfile(path):
            print ("Writing T-learner model to path {}".format(path))
            train_treatment_mask = train_subset['W'].astype(bool)
            X_train = train_subset['X']
            y_train = train_subset['y']
            print('fitting m0')

            self.m0.fit(X_train[~train_treatment_mask,:], y_train[~train_treatment_mask])
            print('fitting m1')
            self.m1.fit(X_train[train_treatment_mask,:], y_train[train_treatment_mask])

            with open(path, 'wb') as handle:
                pickle.dump(self.m0,handle)
            with open(path2, 'wb') as handle:
                pickle.dump(self.m1,handle)

        else:
            print ("Loaded T-learner model from path {}".format(path))
            with open(path, 'rb') as handle:
                self.m0 = pickle.load(handle)

            with open(path2, 'rb') as handle:
                self.m1 = pickle.load(handle)


    def predict(self, X):
        return self.m1.predict(X) - self.m0.predict(X) 




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
        residual: bool = False
    ):
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        assert layer_norm + batch_norm <= 1
        self.residual = residual
        super(MLPLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
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
            regress = False,
            residual = False
    ):
        super(MLPModel, self).__init__()

        self.encoder = MLPLayer(input_dim, model_dim, dropout, batch_norm)
        self.layers = nn.ModuleList(
            [ MLPLayer(model_dim,
                model_dim, dropout, residual=residual) for _ in range(n_layers)]
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
        self.task_encoder = MLPLayer(task_input_dim, model_dim, dropout, batch_norm)
        self.layers = nn.ModuleList(
            [ MLPLayer(model_dim * (2 if i==0 else 1),  # first layer needs to catch double dim
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
            batch_norm = False,
            regress = False,
            residual = False
    ):
        super(MLPLateConcatModelLayernorm, self).__init__()
        # data and task embeddings are encoded separately
        self.encoder = MLPLayer(data_input_dim, model_dim, dropout, layer_norm=True)
        # encoder = data_encoder, but we keep naming for consistent layer name in l1 loss
        self.task_encoder = MLPLayer(task_input_dim, model_dim//2, dropout, layer_norm=True)
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
