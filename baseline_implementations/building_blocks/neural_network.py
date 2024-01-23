import logging

import wandb
from torch import nn
from torch_geometric.data.batch import Batch
from laboratory.building_blocks.utils import get_optimizer_scheduler


class NeuralNetworkEstimator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def train_step(self, device, train_loader, epoch: int, log_interval: int,max_steps:int):
        self.optimizer, self.lr_scheduler = get_optimizer_scheduler(
            args=self.args, model=self
        )

        self.train()
        count_steps = 0
        for batch_idx, batch in enumerate(train_loader):
            print(batch_idx, max_steps)

            if count_steps > max_steps:
                break

            covariates, treatment_node_features, target_outcome = batch
            covariates  = covariates.to(device)
            treatment_node_features  = treatment_node_features.to(device)
            target_outcome  = target_outcome.to(device)

            self.optimizer.zero_grad()

            prediction = self.forward((covariates, treatment_node_features, target_outcome))

            loss = self.loss(prediction, (covariates, treatment_node_features, target_outcome))
            loss.backward()
            self.optimizer.step()
            if batch_idx % log_interval == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * self.args.batch_size,
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
                print({"epoch": epoch, "train_loss": loss.item()})

            count_steps+=1

    def test_prediction(self, batch: Batch):
        return self.forward(batch).view(-1)
