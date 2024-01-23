"""Training routine script"""
from argparse import ArgumentParser, Namespace
import json
from functools import partial
import os
import sys
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import torch
import wandb

sys.path.append(os.getcwd()) # hack for executing module as script (for wandb)

import embeddings.models 


def namespace_without_none(namespace):
    new_namespace = Namespace()
    for key, value in vars(namespace).items():
        if value is not None and type(value) != type:
            if hasattr(value, '__len__'):
                if len(value) == 0:
                    continue
            setattr(new_namespace, key, value)
    return new_namespace


def main(hparams, model_cls):
    """Main function train model."""

    model = model_cls(**vars(namespace_without_none(hparams)))
    
    # Loggers and callbacks
    job_name = f'{hparams.model}_{hparams.dataset}'
    save_dir = f'wandb_output/'
    os.makedirs(save_dir, exist_ok=True)

    wandb_logger = WandbLogger(
        name=job_name,
        project="drug-se-pred",
        entity="dsp", # team name
        log_model=True,
        tags=[
            hparams.model,
            hparams.dataset,
        ],
        save_dir=save_dir,
        #settings=wandb.Settings(start_method="fork")
    )

    monitor_score = hparams.monitor
    monitor_mode = hparams.monitor_mode

    model_checkpoint_cb = ModelCheckpoint(
        monitor=monitor_score,
        mode=monitor_mode,
        save_top_k=1,
        dirpath=wandb_logger.experiment.dir #checkpoint_dir
    )
    early_stopping_cb = EarlyStopping(
        monitor=monitor_score, patience=5, mode=monitor_mode, strict=True,
        verbose=1)

    # most basic trainer, uses good defaults
    trainer = pl.Trainer(
        callbacks=[early_stopping_cb, model_checkpoint_cb],
        #### checkpoint_callback=model_checkpoint_cb,
        max_epochs=hparams.max_epochs,
        logger=wandb_logger,
        gpus=hparams.gpus,
        val_check_interval=0.001,
        ###max_steps=50 #for debugging
        #profiler='advanced'
    )
    trainer.fit(model)
    print('Loading model with', monitor_mode, monitor_score)
    print('Best model path:', model_checkpoint_cb.best_model_path)
    loaded_model = model_cls.load_from_checkpoint(
        checkpoint_path=model_checkpoint_cb.best_model_path)
    # TODO: split and test set not yet implemented, add below once this is ready.
    results = trainer.test(loaded_model)
    # results is a single-element list of a dict:
    results = results[0] #TODO: this assumption correct?

    for name, value in results.items():
        if name in ['labels', 'predictions']:
            continue
        wandb_logger.experiment.summary[name] = value

if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--model', choices=embeddings.models.__all__, type=str,
                        default='GRUAutoencoder')
    parser.add_argument('--max_epochs', default=100, type=int)
    parser.add_argument('--gpus', type=int, default=None)
    parser.add_argument('--hyperparam-draws', default=0, type=int)
    parser.add_argument('--monitor', type=str,
                        default='online_val/loss')
    parser.add_argument('--monitor_mode', type=str, choices=['max', 'min'],
                        default='min')
    parser.add_argument('--dummy_repetition', type=int, default=1,
                        help='inactive argument, used for debugging (enabling grid repetitions)')
   
    # figure out which model to use
    temp_args = parser.parse_known_args()[0]

    if temp_args.monitor.endswith('loss') and temp_args.monitor_mode == 'max':
        print(
            'It looks like you are trying to run early stopping on a loss '
            'using the wrong monitor mode (max).')
        print('Exiting...')
        import sys
        sys.exit(1)

    # let the model add what it wants
    model_cls = getattr(embeddings.models, temp_args.model)

    parser = model_cls.add_model_specific_args(parser)
    hparams = parser.parse_args()
    if isinstance(hparams.dataset, (list, tuple)) and len(hparams.dataset) == 1:
        hparams.dataset = hparams.dataset[0].split(',')

    hparams.dataset_kwargs = {
        #'fold': hparams.rep, # for now inactive                                    
    }
    

    hparams = Namespace(**vars(hparams))
    main(hparams, model_cls)
