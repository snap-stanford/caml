import argparse 
import wandb
from inspect import signature

# TODO: clean up these imports from notebook (many not necessary)
import os
import gzip
import json
import scipy
import random
import math
import numpy as np
import pickle as pkl
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict, KFold, train_test_split
from scipy.sparse import hstack
import pickle
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from consts_and_utils import (parseDataset, getVocab)
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from IPython.core.debugger import Pdb;
from IPython import embed
from pytorch_lightning.loggers import WandbLogger
import sys 
from typing import Union
from scipy.sparse import csr_matrix, coo_matrix
from consts_and_utils import *
import pdb; 
sys.path.append(os.getcwd())

import models
from torch_utils import BestValueLogger, EarlyStopping


def main(args):
    """ Training script for deep models-based for CATE predition 
        All arguments to the main() function are hyperparameters that can be set in the wandb sweep
    """

    run = wandb.init(config=args,
        project="dl-single-task-methotrexate-pancytopenia", 
        entity='dsp'
    )

    logWANDBRun(run)

    # Some preprocessing code first:
    X_labels, Y_labels = getVocab()
    data, subsets, dataset_hash = parseDataset(
        loc="/home/anon/ClaimsCausal/BigQuery/methotrexate_pancytopenia.pickle", 
        trt='DB00563', 
        labels=X_labels, 
        dropEmpty=True
    )
    # Use splits:
    subsets['train']['X'] = subsets['train']['X'].astype(np.float32)
    subsets['val']['X'] = subsets['val']['X'].astype(np.float32)
    subsets['test']['X'] = subsets['test']['X'].astype(np.float32)
    
    X_train = subsets['train']['X']
    X_val = subsets['val']['X']
    X_test = subsets['test']['X']

    y_train = subsets['train']['y']
    y_val = subsets['val']['y']
    y_test = subsets['test']['y']

    train_treatment = subsets['train']['W']
    val_treatment = subsets['val']['W']
    test_treatment = subsets['test']['W']

    X_train_1 = X_train[ subsets['train']['W'].nonzero()[0],:].astype(np.float32)
    X_val_1 = X_val[subsets['val']['W'].nonzero()[0],:].astype(np.float32)
    X_test_1 = X_test[subsets['test']['W'].nonzero()[0],:].astype(np.float32)

    X_train_0 = X_train[(1-subsets['train']['W']).nonzero()[0],:].astype(np.float32)
    X_val_0 = X_val[(1-subsets['val']['W']).nonzero()[0],:].astype(np.float32)
    X_test_0 = X_test[(1-subsets['test']['W']).nonzero()[0],:].astype(np.float32)

    y_train_0 = y_train[(1-subsets['train']['W']).nonzero()[0]]
    y_val_0 = y_val[(1-subsets['val']['W']).nonzero()[0]]
    y_test_0 = y_test[(1-subsets['test']['W']).nonzero()[0]]

    y_train_1 = y_train[subsets['train']['W'].nonzero()[0]]
    y_val_1 = y_val[subsets['val']['W'].nonzero()[0]]
    y_test_1 = y_test[subsets['test']['W'].nonzero()[0]]


    # To tensors:
    y_train_tensor = torch.Tensor(y_train_0)
    y_val_tensor = torch.Tensor(y_val_0)

    # Device:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device {device}')

    full_Y0_hat = None

    if args.step == 'tau':
        model_path = os.path.dirname(os.path.realpath(args.y0_model))
        y0_hat_path = '{}/{}_y0_hat.csv'.format(model_path,dataset_hash)


        # load model and make prediction for everything
        if os.path.isfile(y0_hat_path) == False:
            model = torch.load(args.y0_model)
            model.eval()

            y_tensor_1 = torch.Tensor(np.concatenate((y_train_1, y_val_1)))
            X_tensor_1 = scipy.sparse.vstack([X_train_1, X_val_1])

            full_dataset = ClaimsDataset(X_tensor_1, y_tensor_1)
            full_loader = DataLoader(full_dataset, batch_size = args.batch_size, num_workers=0)
                    
            full_Y0_hat = pd.Series(predict(model, full_loader, device).detach().cpu().numpy()).reset_index()
            full_Y0_hat.columns = ['rowid','y0_hat']
            full_Y0_hat.to_csv(y0_hat_path)

        
        full_Y0_hat = torch.tensor(pd.read_csv(y0_hat_path)['y0_hat'])


    # Torch dataset:
    train_dataset = ClaimsDataset(X_train_0,y_train_tensor, y_train_tensor*args.positive_weight)
    val_dataset = ClaimsDataset(X_val_0,y_val_tensor, y_val_tensor*args.positive_weight) 

    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size = args.batch_size, num_workers=0) 

    if args.step == 'tau':        
        train_tau_label_1 =   (torch.Tensor(y_train_1) - full_Y0_hat[0:X_train_1.shape[0]]).float()
        val_tau_label_1 =   (torch.Tensor(y_val_1) - full_Y0_hat[X_train_1.shape[0]:]).float()

        #pdb.set_trace() 
        
        train_dataset = ClaimsDataset(X_train_1,train_tau_label_1, torch.Tensor(y_train_1) * args.positive_weight)
        val_dataset = ClaimsDataset(X_val_1,val_tau_label_1, torch.Tensor(y_val_1) * args.positive_weight) 

        train_loader = DataLoader(train_dataset, batch_size = args.batch_size, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size = args.batch_size, num_workers=0) 

        y_val_tensor = val_tau_label_1


    #pdb.set_trace() 

    # Model:
    input_dim = subsets['train']['X'].shape[1]
    output_dim = 1

    #model = models.MLPLogistic(
    #    input_dim,
    #    output_dim
    #)
    model = models.MLPModel(
        input_dim, 
        output_dim,
        args.dropout,
        args.model_dim,
        args.n_layers,
        args.batch_norm,
        regress = (args.step == 'tau')
        ) 

    # TODO: getattr(models, model) init with model specific params
    model = nn.DataParallel(model)
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Training Model with {total_params} parameters.')
    print(model)
    
    # Training setup:
    criterion = nn.BCELoss() 
    if args.step == 'tau':
        criterion = nn.MSELoss() 
        if args.loss == 'mae':
            criterion = nn.L1Loss() 
        if args.loss == 'huber':
            criterion = nn.HuberLoss(delta=0.5) 

    #optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)  
    optimizer = get_optimizer(model, args.learning_rate, args.weight_decay) 
    torch.cuda.empty_cache()
    if args.use_lr_scheduler:
        print('Using LR scheduler')
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, 
                max_lr=args.learning_rate, 
                steps_per_epoch=len(train_loader), 
                epochs=args.n_epochs)

    # Init best value logger:
    metrics = [['val_loss', False]]
    if args.step=='y0': #TODO: make this less hard-coded
        metrics.append(['val_roc_auc', True])
    elif args.step=='tau':
        metrics.append(['val_recall_at_99', True])
    else:
        raise ValueError(f' Step {step} not among [tau, y0]')
    names, increasing = list(zip(*metrics)) #unzip
    bv_log = BestValueLogger(list(names), list(increasing))
    print(f'Logging best values of {names}')

    # Init Early Stopping:
    early_stopping = EarlyStopping('val_loss', patience=10) 

    # Train loop:
    for epoch in range(args.n_epochs):
    
        Train_Y0_loss_epoch = 0
        total_train_samples = 0
        print(f'>>>>>> Epoch {epoch}')
        # Train step: 
        for i, (samples, labels, weights) in enumerate(tqdm(train_loader)):
            break
            weights = weights.to(device)
            samples = samples.to(device)
            labels = labels.to(device)
            # Clear gradients w.r.t. parameters
            model.train()
            
            Y0 = model(samples)  

            if args.positive_weight == 1:
                loss = criterion(Y0.view(-1), labels)
            else:
                loss = (weights * (Y0.view(-1) - labels) ** 2).mean()

            l1_term = model.module.encoder.layer[0].weight.abs().sum()
            l1_term = l1_term * (4096. / args.batch_size) # make l1 term independent from bsz.
            print(f'l1: {l1_term}, loss: {loss}, ratio: {l1_term/loss}')
            loss = loss + args.l1_reg*l1_term  
            Y0_loss = loss.item()
            
            Train_Y0_loss_epoch += Y0_loss* samples.size(0)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_train_samples+=samples.size(0)
            if args.use_lr_scheduler:
                scheduler.step()
           
        # Val Step:
        model.eval()
        
        val_Y0 = []
            
        with torch.no_grad():
            print('start eval')
            #print('loader size {}'.format(len(val_loader)))
            counter = 0
            
            for ix, (samples, labels, weights) in enumerate(tqdm(val_loader)):            
                counter+=1
                weights = weights.to(device)
                samples = samples.to(device)
                labels = labels.to(device)
                Y0 = model(samples).view(-1)
                val_Y0.append(Y0)
                
        val_Y0 = torch.cat(val_Y0)

        if args.positive_weight == 1:
            val_loss_y0 = criterion(val_Y0, y_val_tensor.to(device))
        else:
            val_loss_y0 = ((torch.Tensor(y_val_1) * args.positive_weight).to(device) * (val_Y0 - y_val_tensor.to(device)) ** 2).mean()

        train_loss = Train_Y0_loss_epoch/total_train_samples

        logInfo = {"epoch": epoch, "val_loss": val_loss_y0.cpu(),"train_loss": train_loss}
        if args.use_lr_scheduler:
            logInfo.update(
                {'learning_rate': scheduler.get_last_lr()[0]}
            )
        
        current_vals = [logInfo['val_loss']]
        if args.step=='y0':
            logInfo[ "val_roc_auc"] = roc_auc_score( y_val_tensor.cpu(), val_Y0.cpu().numpy())
            current_vals.append( logInfo["val_roc_auc"])
        if args.step=='tau':
            logInfo[ "val_recall_at_99"] = recallAt( y_val_1, val_Y0.cpu(), 0.99)
            current_vals.append( logInfo["val_recall_at_99"])

        # update best values:
        bv_log(current_vals, epoch) #TODO: update per eval step (which may be more than 1x/epoch)
        # perform early stopping:
        if early_stopping([logInfo['val_loss']], epoch):
            print(f'Stopping training early after epoch {epoch}.')
            break

        print(logInfo)
        wandb.log(logInfo)
        
        torch.save(model, os.path.join(getWANDBDir(run), "model_{}.pt".format(epoch)))
        wandb.save(os.path.join(getWANDBDir(run), "model_{}.pt".format(epoch)))

    # Log best metrics:
    wandb.log(bv_log.best_values) 
 

if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--y0_model', type=str, default='none')
    parser.add_argument('--step', choices=['tau','y0'], type=str, default='y0')
    parser.add_argument('--model', choices=models.__all__, type=str,
                        default='MLPModel')
    parser.add_argument('--dataset', default='DB00563-Pancytopenia')
    # Training config:
    parser.add_argument('--n_epochs', type=int,
                        default=100)
    parser.add_argument('--learning_rate', type=float,
                        default=0.0003)
    parser.add_argument('--batch_size', type=int,
                        default=2048*4)
    parser.add_argument('--l1_reg', type=float,
                        default=1e-7) # of the encoder layer
    parser.add_argument('--use_lr_scheduler', type=str,
                        default="false") # hack to cope with wandb 
    # Model params:
    parser.add_argument('--n_layers', type=int,
                        default=2)
    parser.add_argument('--model_dim', type=int,
                        default=256)
    parser.add_argument('--batch_norm', type=str, default="false") # hack to cope with wandb
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--loss', choices=['mse','huber','mae'], type=str,
                        default='mse')
    parser.add_argument('--positive_weight', type=int,
                        default=1)

    args = parser.parse_args()
    # update pseudo-bool arguments:
    args = convert_str_to_bool(args, ['batch_norm', 'use_lr_scheduler'])
    print(f'Boolean args: {args.batch_norm=}, {args.use_lr_scheduler=}')
    main(args) 


