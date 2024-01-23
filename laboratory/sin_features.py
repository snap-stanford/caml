import logging
from argparse import Namespace

import torch.nn.functional as F
import wandb
from torch import Tensor, nn
from torch_geometric.data.batch import Batch

from laboratory.building_blocks.como import COMONet
from laboratory.building_blocks.covariates_feature_extractor import \
    CovariatesFeatureExtractor
from laboratory.building_blocks.propensity_feature_extractor import PropensityNet
from laboratory.building_blocks.treatment_feature_extractor import \
    TreatmentFeatureExtractorMLP
from laboratory.building_blocks.utils import get_optimizer_scheduler


class SIN(nn.Module):
    def __init__(self, args: Namespace):
        super(SIN, self).__init__()
        self.treatment_net = TreatmentFeatureExtractorMLP(args=args)
        self.propensity_net = PropensityNet(args=args)
        self.como_net = args.como_net
        self.covariates_net = CovariatesFeatureExtractor(args=args)

        self.treatment_net_opt, self.treatment_net_lr_sched = get_optimizer_scheduler(
            args=args, model=self.treatment_net, net="gnn"
        )
        self.propensity_net_opt, self.propensity_net_lr_sched = get_optimizer_scheduler(
            args=args, model=self.propensity_net, net="propensity"
        )
        self.como_net_opt, self.com_net_lr_sched = get_optimizer_scheduler(
            args=args, model=self.como_net, net="com"
        )
        self.covariates_net_opt, self.covariates_net_lr_sched = get_optimizer_scheduler(
            args=args, model=self.covariates_net
        )
        #self.is_multi_relational = args.gnn_multirelational

        self.num_update_steps_como = args.num_update_steps_como
        self.num_update_steps_propensity = args.num_update_steps_propensity
        self.num_update_steps_global_objective = args.num_update_steps_global_objective
        self.log_interval = args.log_interval

    def eval(self):
        self.treatment_net.eval()
        self.propensity_net.eval()
        self.como_net.eval()
        self.covariates_net.eval()

    def train(self, mode: bool = True):
        self.treatment_net.train(mode)
        self.propensity_net.train(mode)
        self.como_net.train(mode)
        self.covariates_net.train(mode)

    def com_loss(self, prediction: Tensor, target: Tensor):
        return F.mse_loss(input=prediction.view(-1), target=target)

    def propensity_loss(self, prediction: Tensor, target: Tensor):
        return F.mse_loss(input=prediction, target=target)

    def global_loss(self, prediction: Tensor, target: Tensor):
        return F.mse_loss(input=prediction.view(-1), target=target)

    def test_prediction(self, batch: Batch):
        return self.forward(batch).view(-1)

    def forward(self, batch: Batch):
       # (
       #     treatment_node_features,
       #     covariates,
       #     target_outcome,
       # ) = (batch.x, batch.covariates, batch.y)
       
        covariates, treatment_node_features, target_outcome = batch

        #from IPython.core.debugger import Pdb; Pdb().set_trace()

        treatment_node_features = self.treatment_net(
            treatment_node_features,
        )
        covariates_features = self.covariates_net(covariates)
        propensity_features = self.propensity_net(covariates)
        com = self.como_net(covariates).view(-1)
        diff = treatment_node_features - propensity_features
        Y = (diff * covariates_features).sum(-1).view(-1) + com
        return Y

    def train_step(self, device, train_loader, epoch: int, log_interval: int,max_steps:int):
        self.train()
        count_steps = 0
        for batch_idx, batch in enumerate(train_loader):
            print(batch_idx)

            if count_steps > max_steps:
                break
            count_steps+=1
            #batch = batch.to(device)
            #(
            #    treatment_node_features,
            #    covariates,
            #    target_outcome,
            #) = (batch.x, batch.covariates, batch.y)

            covariates, treatment_node_features, target_outcome = batch
            covariates  = covariates.to(device)
            treatment_node_features  = treatment_node_features.to(device)
            target_outcome  = target_outcome.to(device)

            for _ in range(self.num_update_steps_global_objective):
                self.covariates_net_opt.zero_grad()
                self.treatment_net_opt.zero_grad()
                global_prediction = self.forward((covariates, treatment_node_features, target_outcome))

                global_loss = self.global_loss(
                    prediction=global_prediction, target=target_outcome
                )
                global_loss.backward()
                self.covariates_net_opt.step()
                self.treatment_net_opt.step()

            for _ in range(self.num_update_steps_propensity):
                treatment_features = self.treatment_net(
                    treatment_node_features,
                )
                self.propensity_net_opt.zero_grad()
                propensity_features = self.propensity_net(covariates)
                propensity_loss = self.propensity_loss(
                    prediction=propensity_features, target=treatment_features
                )
                propensity_loss.backward()
                self.propensity_net_opt.step()

            if batch_idx % log_interval == 0:
                print(
                    f"Train Epoch: {epoch} [{batch_idx * len(batch)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\t Global Objective Loss: {global_loss.item():.6f},\t Propensity loss: {propensity_loss.item()}"
                )

        self.treatment_net_lr_sched.step()
        self.propensity_net_lr_sched.step()
        self.covariates_net_lr_sched.step()
