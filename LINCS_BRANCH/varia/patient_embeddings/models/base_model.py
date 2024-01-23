"""Base model to learn representation of patient history."""
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl

from sklearn.metrics import (
    average_precision_score, roc_auc_score, balanced_accuracy_score)

from embeddings.utils import (
    variable_length_collate)
import embeddings.datasets

class BaseModel(pl.LightningModule):
    """Base model for embedding models."""

    @property
    def metrics_initial(self):
        return {
            'train/loss': float('inf'),
            'online_val/loss': float('inf'),
            'validation/loss': float('inf'),
        }

    def __init__(self, dataset_kwargs={}, **kwargs): #dataset, learning_rate, batch_size, weight_decay, 
        super().__init__()
        self.save_hyperparameters()

        # get dataset class and initialize it:
        self.dataset_cls = getattr(embeddings.datasets, self.hparams.dataset)
        d = self.dataset_cls(split='train', **dataset_kwargs)
        # get online split of the training split, used for early stopping:
        self.train_indices, self.val_indices = d.get_split(87346583)
        self.dataset_kwargs = dataset_kwargs
        self.loss = torch.nn.MSELoss(reduction='none')

    def training_step(self, batch, batch_idx):
        """Run a single training step."""
        data, lengths = batch['data'], batch['lengths']
        output = self.forward(data, lengths) #.squeeze(-1)
        loss = self.loss(output, data)

        total_loss = loss.mean()
        self.log('loss', total_loss)
        return {
            'loss': total_loss,
        }

    def training_epoch_end(self, outputs):
        total_loss = 0
        for x in outputs:
            # currently: macro average of loss over all batches 
            # (irrespective of time series length)
            total_loss += x['loss']
       
        # n_total approximated from batch_size * n_iters in epoch
        # TODO: extract exact n directly from dataset class
        batch_size = self.hparams.batch_size
        n_total = batch_size * len(outputs) 
        # TODO: could also consider time point wise aggregation of loss 
        
        average_loss = total_loss / n_total  
        self.log('train/loss', average_loss, prog_bar=True)

    def _shared_eval(self, batch, batch_idx, prefix):
        """ Evaluation step """
        data, lengths = (
            batch['data'],
            batch['lengths'],
        )
        output = self.forward(data, lengths) #.squeeze(-1)

        loss = self.loss(output, data)
        total_loss = loss.mean()

        return {
            f'{prefix}_loss': total_loss.detach(),
            #f'{prefix}_scores': output.cpu().detach().numpy()
        }

    def _shared_end(self, outputs, prefix):
        total_loss = 0
        loss_key = f'{prefix}_loss'
        for x in outputs:
            total_loss += x[loss_key]
       
        batch_size = self.hparams.batch_size
        n_total = batch_size * len(outputs) 
        # TODO: could also consider time point wise aggregation of loss 
        
        average_loss = total_loss / n_total  
        self.log(f'{prefix}/loss', average_loss, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, 'online_val')

    def validation_epoch_end(self, outputs):
        self._shared_end(outputs, 'online_val')

    def test_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, prefix='validation')

    def test_epoch_end(self, outputs):
        self._shared_end(outputs, 'validation')

    def configure_optimizers(self):
        """Get optimizers."""
        # TODO: We should also add a scheduler here to implement warmup. Most
        # recent version of pytorch lightning seems to have problems with how
        # it was implemented before.

        # here we don't apply weight decay to bias and layernorm parameters, as inspired by:
        # https://colab.research.google.com/github/PytorchLightning/pytorch-lightning/blob/master/notebooks/04-transformers-text-classification.ipynb
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ] 
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        return optimizer

    def train_dataloader(self):
        """Get train data loader."""
        return DataLoader(
            Subset(
                self.dataset_cls(
                    split='train',
                    **self.dataset_kwargs
                ),
                self.train_indices
            ),
            shuffle=True,
            collate_fn=variable_length_collate,
            batch_size=self.hparams.batch_size,
            num_workers=1,
            pin_memory=True
        )

    def val_dataloader(self):
        """Get validation data loader."""
        return DataLoader(
            Subset(
                self.dataset_cls(
                    split='train',
                    **self.dataset_kwargs
                ),
                self.val_indices
            ),
            shuffle=False,
            collate_fn=variable_length_collate,
            batch_size=self.hparams.batch_size,
            num_workers=1,
            pin_memory=True
        )

    def test_dataloader(self):
        """Get test data loader."""
        return DataLoader(
            self.dataset_cls(
                split='validation',
                **self.dataset_kwargs
            ),
            shuffle=False,
            collate_fn=variable_length_collate,
            batch_size=self.hparams.batch_size,
            num_workers=1
        )

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        """Specify the hyperparams."""
        parser = argparse.ArgumentParser(parents=[parent_parser])
        # training specific
        parser.add_argument(
            '--dataset',
            default='MPDataset', choices=embeddings.datasets.__all__
        )
        parser.add_argument(
            '--learning_rate', default=0.001, type=float,
            # tunable=True, log_base=10., low=0.0001, high=0.01
        )
        parser.add_argument(
            '--batch_size', default=32, type=int,
            # options=[16, 32, 64, 128, 256], tunable=True
        )
        parser.add_argument('--weight_decay', default=0., type=float)
        return parser
