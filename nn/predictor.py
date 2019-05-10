#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : nn/predictor.py
# Author            : Hai-Yong Jiang <haiyong.jiang1990@hotmail.com>
# Date              : 19.08.2018
# Last Modified Date: 10.05.2019
# Last Modified By  : Hai-Yong Jiang <haiyong.jiang1990@hotmail.com>
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, MultiStepLR
import torch.optim as optim
import tensorflow as tf
import copy
import numpy as np
import GPUtil

class Predictor(object):
    def __init__(self, cfg, net):
        self.cfg = cfg
        self.net = net
        self.max_epochs = cfg.OPTM.EPOCHS
        self.epoch_counter = 0

    def print_net_params(self):
        tf.logging.info("Network architecture: ")
        for k in self.net.state_dict:
            tf.logging.info("%s(%s)"%(k, str(self.net.state_dict[k])))

    def restore_model(self, model_path):
        """ Restore a model parameters from the one given in argument

        params:
            @model_path: if string, directly restore; if a dict, add a header.
        """
        tf.logging.info("Load state_dicts: ")
        if isinstance(model_path, str):
            state_dict_ref = torch.load(model_path)
            self.net.load_state_dict(state_dict_ref)
            tf.logging.info("\n".join(["%s: %s"%(k,str(v.shape))
                                        for k,v in state_dict_ref.items()]))
        elif isinstance(model_path, dict):
            state_dict = {}
            for key in model_path:
                state_dict_ref = torch.load(model_path[key])
                state_dict.update({key+"."+k:v for k,v in state_dict_ref.items()})
                tf.logging.info("\n".join(["%s: %s"%(k,str(v.shape))
                                           for k,v in state_dict_ref.items()]))
            self.net.load_state_dict(state_dict, strict=False)

    def _criterion(self, preds, targets):
        raise NotImplementedError('subclasses must override _criterion()!')
        return 0.0

    def _metric(self, preds, targets, ndim=3, name="vt"):
        raise NotImplementedError('subclasses must override _criterion()!')

    def _register_params(self, sample):
        state_dict = self.net.state_dict(keep_vars=True)
        for k in state_dict:
            if k.endswith("weight"):
                sample["param_"+k] = state_dict[k].data.cpu().numpy()
                sample["grad_"+k] = state_dict[k].grad.data.cpu().numpy()
                sample["update_"+k] = np.abs(sample["grad_"+k])/(1e-8+np.abs(sample["param_"+k]))

    def _update_metrics(self, metrics, loss, metric):
        """
        Update given losses and metrics. The update is in-place

        params:
            @metrics: a dict, the summary of metrics
            @loss: a dict, the present measured loss
            @metric, a dict or list the present measured metric.
        """
        for k,v in loss.items():
            metrics.setdefault(k, 0)
            metrics[k] = metrics[k] + loss[k].item()

        assert(isinstance(metric, dict))
        for k,v in metric.items():
            if k in metrics:
                metrics[k] = metric[k] + metrics[k]
            else:
                metrics.setdefault(k, v)

            if "min" in k:
                metrics[k] = min(metric[k], metrics[k])
            elif "max" in k:
                metrics[k] = max(metric[k], metrics[k])

        return metrics

    def _normalize_metrics(self, metrics, nsample):
        """
        Normalize the items in metrics with a size, nsample. The normalization is in-place.
        """
        for k in metrics:
            if "min" not in k and "max" not in k:
                metrics[k] /= nsample
        return metrics

    def _train_epoch(self, train_loader, optimizer, epoch):
        raise NotImplementedError('subclasses must override _criterion()!')

    def _validate_epoch(self, val_loader):
        raise NotImplementedError('subclasses must override _criterion()!')

    def get_params(self, layers="all"):
        params = self.net.named_parameters()
        if layers == "all":
            params = {k:v for k,v in params}
        else:
            params = {k:v for k,v in params if k.startswith(layers)}
        print("Selected params for %s: "%layers)
        print(",".join(params.keys()))
        return params.values()

    def setup_optimizer(self, layers="all"):
        if isinstance(layers, str):
            optimizer = optim.Adam(self.get_params(layers=layers), lr = self.cfg.OPTM.LR)
        elif isinstance(layers, dict):
            param_list = []
            for name,lr in layers.items():
                param_list.append({"params": self.get_params(name), "lr": lr})
            optimizer = optim.Adam(param_list, lr = self.cfg.OPTM.LR)
        return optimizer

    def train(self, train_loader, valid_loader, epochs, callbacks=None,
              layers="all"):
        """
            Trains the neural net
        Args:
            train_loader (DataLoader): The Dataloader for training
            valid_loader (DataLoader): The Dataloader for validation
            epochs (int): number of epochs
            threshold (float): The threshold used to consider the mask present or not
            callbacks (list): List of callbacks functions to call at each epoch
            layers: a string (the layer name) or a dict (layer name and its learning rate)
        Returns:
            str, None: The path where the model was saved, or None if it wasn't saved
        """
        optimizer = self.setup_optimizer(layers=layers)
        if self.cfg.OPTM.LR_SCHEDULER=="ReduceLROnPlateau":
            lr_scheduler = ReduceLROnPlateau(optimizer, 'min',
                                             **self.cfg.OPTM.LR_SCHEDULER_PARAM)
        else:
            lr_scheduler = MultiStepLR(optimizer,
                                       **self.cfg.OPTM.LR_SCHEDULER_PARAM)

        states = {"lr": optimizer.param_groups[0]["lr"],
                  "state_dict": [],
                  "val_loss": 0.0}

        for epoch in range(epochs):
            ## print gpu information
            print("GPU usage: ")
            GPUtil.showUtilization()

            tf.logging.info("\n\nEpoch %d, learning rate: %e, %s"%(epoch,
                            optimizer.param_groups[0]["lr"], self.cfg.MODEL.IDENTIFIER) )
            ## switch to train mode
            self.net.train()
            ## Run a train pass on the current epoch
            train_loss, train_sample = self._train_epoch(train_loader, optimizer, epoch)

            # switch to evaluate mode
            self.net.eval()
            # Run the validation pass
            with torch.no_grad():
                val_loss, val_sample = self._validate_epoch(valid_loader)

            ## save best model
            if epoch==0 or states["val_loss"]>=val_loss["loss_sum"]:
                states["val_loss"] = val_loss["loss_sum"]
                states["val_metric"] = val_loss
                states["train_metric"] = train_loss
                states["state_dict"] = copy.deepcopy(self.net.state_dict())

            # Reduce learning rate if needed
            if self.cfg.OPTM.LR_SCHEDULER=="ReduceLROnPlateau":
                lr_scheduler.step(val_loss["loss_sum"])
            else:
                lr_scheduler.step()

            if states["lr"] > optimizer.param_groups[0]["lr"]:
                ## restore the model
                tf.logging.info("Restore the best model.")
                self.net.load_state_dict(states["state_dict"])
                states["lr"] = optimizer.param_groups[0]["lr"]

            # If there are callback call their __call__ method and pass in some arguments
            if callbacks:
                for cb in callbacks:
                    cb(step_name="epoch",
                    net=self.net,
                    train_sample=train_sample,
                    val_sample=val_sample,
                    epoch_id=epoch,
                    train_loss= train_loss,
                    val_loss=val_loss,
                    lr = optimizer.param_groups[0]["lr"]
                    )

            if optimizer.param_groups[0]["lr"] <= 1e-7:
                break

        print("Best validation results:")
        print("train loss: ", states["train_metric"])
        print("val loss: ", states["val_metric"])
        # If there are callback call their __call__ method and pass in some arguments
        if callbacks:
            for cb in callbacks:
                cb(step_name="train", net=self.net, epoch_id=epoch)
        torch.save(states["state_dict"], self.cfg.DATA.RES_MODEL_PATH.replace(".pth", "_best.pth"))


