""" caml classes """

from tqdm import tqdm
import torch
import gc
import pdb
from pdb import set_trace as bp
from laboratory.building_blocks.hsic_utils import hsic_normalized

def print_memory():
    x = torch.cuda.memory_allocated()
    print(f'CUDA MEMORY USAGE : {x/(10**9)} GB')


class camlAdapter():
    """
    Class to encapsulate the inner loop of the caml algorithm.
    """
    def __init__(self, loss_fn, adaptation_steps, eval_steps, device, args,propensity_loss=None):
        self.loss_fn = loss_fn
        self.adaptation_steps = adaptation_steps
        self.eval_steps = eval_steps
        self.device = device
        self.args = args # argparse args
        self.propensity_loss = propensity_loss

    def _get_pred(self, batch, model):
        device = self.device
        if self.args.task_embedding_config in ['late_concat', 'late_concat_layernorm', 'hypernet']:
            data, task_embedding, labels = batch
            data, task_embedding, labels = data.to(device),\
                task_embedding.to(device), labels.to(device)
            tau_pred = model(data, task_embedding)
        else:
            data, labels = batch
            data, labels = data.to(device), labels.to(device)
            tau_pred = model(data)
        return tau_pred, labels

    def __call__(self, train_loader, val_loader, model, adapt_opt,other_opt=None,flatten=True):
        """
        Adapts the provided model for <adaptation_steps> many batches
        from the train_loader. Evaluates loss on <eval_steps> many
        batches of the val_loader, returns average val_loss
        """
        device = self.device
        args = self.args

        # Adapt the model
        model.train()
        for inner_iter, batch in enumerate(tqdm(train_loader)):

            #break
            #pdb.set_trace()
            print(f'Inner Train loop: {inner_iter}')
            if inner_iter + 1 > self.adaptation_steps:
                break

            #adapt_opt.zero_grad()
            # print(f'Before train _get_pred: {inner_iter}'); print_memory()
            if self.args.sin_learner == True:
                covariates, treatment_node_features, labels = batch
                covariates  = covariates.to(device)
                treatment_node_features  = treatment_node_features.to(device)
                labels  = labels.to(device)

            elif self.args.graphite_learner == True:
                covariates, treatment_node_features, labels = batch
                covariates  = covariates.to(device)
                treatment_node_features  = treatment_node_features.to(device)
                labels  = labels.to(device)
                tau_pred, covariates_features, treatment_features = model((covariates, treatment_node_features, labels))

            elif self.args.task_embedding_config in ['late_concat', 'late_concat_layernorm', 'hypernet']:
                #if self.args.caml_r_learner == False:
                data, task_embedding, labels = batch
                #else:
                #data, task_embedding, labels, weights = batch
                    


                if self.args.null_patients2:
                    null_embedding = torch.zeros_like(task_embedding)
                    data = torch.cat([data,data])
                    task_embedding = torch.cat([task_embedding, null_embedding])
                    labels = torch.cat([labels, torch.zeros_like(labels)])

                    #if self.args.caml_r_learner:
                    #    weights = torch.cat([weights, weights])
                    #    weights = weights.to(device)

                data  = data.to(device)
                task_embedding = task_embedding.to(device)
                labels = labels.to(device)
                tau_pred = model(data, task_embedding)
                task_embedding.detach()

                                                

            else:
                data, labels = batch
                data = data.to(device)
                labels = labels.to(device)
                tau_pred = model(data)


            if self.args.sin_learner == True:
                for _ in range(self.args.num_update_steps_global_objective):
                    other_opt['covariates_net_opt'].zero_grad()
                    other_opt['treatment_net_opt'].zero_grad()
                    global_prediction = model.forward((covariates, treatment_node_features, labels))
                    global_loss = model.global_loss(
                        prediction=global_prediction, target=labels
                    )
                    global_loss.backward()
                    other_opt['covariates_net_opt'].step()
                    other_opt['treatment_net_opt'].step()


                for _ in range(self.args.num_update_steps_propensity):
                    treatment_features = model.treatment_net(
                        treatment_node_features,
                    )
                    other_opt['propensity_net_opt'].zero_grad()
                    propensity_features = model.propensity_net(covariates)
                    propensity_loss = model.propensity_loss(
                        prediction=propensity_features, target=treatment_features
                    )
                    propensity_loss.backward()
                    other_opt['propensity_net_opt'].step()

                print(
                    f"Train Epoch: [{inner_iter * len(batch)}/{len(train_loader.dataset)} ({100. * inner_iter / len(train_loader):.0f}%)]\t Global Objective Loss: {global_loss.item():.6f},\t Propensity loss: {propensity_loss.item()}"
                )
                


            else:

                if flatten:
                    loss = self.loss_fn(tau_pred, labels)
                else:
                    #if args.caml_r_learner:
                    #    loss_fn = torch.nn.MSELoss(reduction='none')
                    #    try:
                    #        loss = (loss_fn(tau_pred, labels) * weights).mean()
                    #    except:
                    #        breakpoint()
                    #    #loss = self.loss_fn(tau_pred.view(-1), labels) * weights
                    #else:
                    loss = self.loss_fn(tau_pred, labels) 


                    loss = self.loss_fn(tau_pred, labels)
                # Regularization:
                # two cases: regular MLP or Hypernetwork: 
                if hasattr(model, 'encoder'): 
                    l1_term = model.encoder.layer[0].weight.abs().sum()
                elif hasattr(model, 'covariates_net'): 
                    l1_term = model.covariates_net.covariates_net.layers[0].weight.abs().sum()
                elif hasattr(model, 'mnet_base'):
                    if hasattr(model.mnet_base, 'encoder'):
                        l1_term = model.mnet_base.encoder.layer[0].weight.abs().sum()
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
    
                loss = loss + args.l1_reg*l1_term
    
                if hasattr(model, 'covariates_net') and model.independence_regularisation_coeff>0: 
                    HSIC_regularisation = model.independence_regularisation_coeff * hsic_normalized(
                        x=treatment_node_features, y=treatment_features
                    )
                    loss = loss + HSIC_regularisation
                # Gradient step:
                loss.backward()
                adapt_opt.step()
                adapt_opt.zero_grad()


                #





            ###
            ### ADDED BREAK
            ###

        if self.args.sin_learner == True:
            return torch.tensor([0.0])

        model.eval()



        # Evaluate adapted model:
        with torch.no_grad():

            val_loss = 0.0
            n_samples = 0


            for eval_iter, batch in enumerate(tqdm(val_loader)):
                print(f'Inner Eval loop: {inner_iter}')


                if eval_iter + 1 > self.eval_steps:
                    break

                if self.args.graphite_learner == True:
                    covariates, treatment_node_features, labels = batch
                    covariates  = covariates.to(device)
                    treatment_node_features  = treatment_node_features.to(device)
                    labels  = labels.to(device)
                    tau_pred, covariates_features, treatment_features = model((covariates, treatment_node_features, labels))

                elif self.args.task_embedding_config in ['late_concat', 'late_concat_layernorm', 'hypernet']:
                    data, task_embedding, labels = batch
                    data  = data.to(device)
                    task_embedding = task_embedding.to(device)
                    labels = labels.to(device)
                    tau_pred = model(data, task_embedding)
                    task_embedding.detach()
                else:
                    data, labels = batch
                    data = data.to(device)
                    labels = labels.to(device)
                    tau_pred = model(data)

                ## data, labels = data.to(device), labels.to(device)
                # print(f'Before eval _get_pred: {inner_iter}'); print_memory()
                ## tau_pred, labels = self._get_pred(batch, model)
                if flatten:
                    val_loss_batch = self.loss_fn(tau_pred, labels)
                else:
                    val_loss_batch = self.loss_fn(tau_pred, labels)
                #pdb.set_trace()
                val_loss += val_loss_batch*batch[0].size(0)
                n_samples += batch[0].size(0)


            if n_samples == 0:
                return torch.tensor([0.0])

            val_loss /= n_samples

            return val_loss
