#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : nn/predictor.py
# Author            : Hai-Yong Jiang <haiyong.jiang1990@hotmail.com>
# Date              : 19.08.2018
# Last Modified Date: 15.09.2019
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
        self.mode = "train"

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
            flag = k in metrics
            metrics.setdefault(k, v)

            if isinstance(v, dict):
                for k1 in v:
                    if k1 not in metrics[k]:
                        metrics[k][k1] = metric[k][k1]
                    else:
                        metrics[k][k1] += metric[k][k1]
            elif "_min" in k:
                metrics[k] = min(metric[k], metrics[k])
            elif "_max" in k:
                metrics[k] = max(metric[k], metrics[k])
            elif "_avg" in k or "_sum" in k:
                if flag:
                    metrics[k] = metric[k] + metrics[k]
            else:
                if flag:
                    metrics[k] = metric[k] + metrics[k]

        return metrics

    def _normalize_metrics(self, metrics, nsample):
        """
        Normalize the items in metrics with a size, nsample. The normalization is in-place.
        """
        for k in metrics:
            if "_avg" in k or "loss_" in k:
                metrics[k] /= nsample
        return metrics

    #  def _train_epoch(self, train_loader, optimizer, epoch):
    #      raise NotImplementedError('subclasses must override _criterion()!')

    #  def _validate_epoch(self, val_loader):
    #      raise NotImplementedError('subclasses must override _criterion()!')

    def get_params(self, layers="all"):
        params = self.net.named_parameters()
        if layers == "all":
            params = {k:v for k,v in params}
        else:
            params = {k:v for k,v in params if k.startswith(layers)}
        print("Selected params for %s: "%layers)
        print(",".join(params.keys()))
        return params.values()

    def get_optimizer(self):
        if self.cfg.OPTM.OPTIMIZER == "ADAM":
            return lambda x,y: optim.Adam(x, y)
        elif self.cfg.OPTM.OPTIMIZER == "SGD":
            return lambda x,y: optim.SGD(x, y, momentum=0.9, weight_decay=0.0005)

    def setup_optimizer(self, layers="all"):
        optimizer = self.get_optimizer()
        if isinstance(layers, str):
            optimizer = self.get_optimizer()(self.get_params(layers=layers), self.cfg.OPTM.LR)
        elif isinstance(layers, dict):
            param_list = []
            for name,lr in layers.items():
                param_list.append({"params": self.get_params(name), "lr": lr})
            optimizer = self.get_optimizer()(param_list, self.cfg.OPTM.LR)
        return optimizer

    def get_largest_lr(self, opt):
        return np.max([v["lr"] for v in opt.param_groups])

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
        elif self.cfg.OPTM.LR_SCHEDULER=="StepLR":
            p_dict = self.cfg.OPTM.LR_SCHEDULER_PARAM
            lr_scheduler = StepLR(optimizer, p_dict["step_size"], gamma=p_dict["gamma"])
        else:
            lr_scheduler = MultiStepLR(optimizer,
                                       **self.cfg.OPTM.LR_SCHEDULER_PARAM)

        states = {"lr": self.get_largest_lr(optimizer),
                  "state_dict": [],
                  "val_loss": 0.0}

        for epoch in range(epochs):
            self.cfg.MODEL.iEPOCH = epoch
            ## print gpu information
            print("GPU usage: ")
            GPUtil.showUtilization()

            tf.logging.info("\n\nEpoch %d, learning rate: %e, %s"%(epoch,
                             self.get_largest_lr(optimizer), self.cfg.MODEL.IDENTIFIER) )
            ## switch to train mode
            self.net.train()
            ## Run a train pass on the current epoch
            train_loss, train_sample = self._train_epoch(train_loader, optimizer, epoch)

            # switch to evaluate mode
            self.net.eval()
            # Run the validation pass
            if epoch%self.cfg.MODEL.VAL_EPOECH_FREQ == 0:
                with torch.no_grad():
                    val_loss, val_sample = self._validate_epoch(valid_loader)
            elif epoch == 0:
                val_loss = {"loss_sum": 1e6}
                val_sample = []

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

            lr_opt = self.get_largest_lr(optimizer)
            if states["lr"] > lr_opt:
                ## restore the model
                tf.logging.info("Restore the best model.")
                self.net.load_state_dict(states["state_dict"])
                states["lr"] = lr_opt

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
                    lr = lr_opt
                    )

            if lr_opt <= 1e-7:
                break

        print("Best validation results:")
        print("train loss: ", states["train_metric"])
        print("val loss: ", states["val_metric"])
        # If there are callback call their __call__ method and pass in some arguments
        if callbacks:
            for cb in callbacks:
                cb(step_name="train", net=self.net, epoch_id=epoch)
        torch.save(states["state_dict"], self.cfg.DATA.RES_MODEL_PATH.replace(".pth", "_best.pth"))
    def _pack_data(self, data):
        raise NotImplementedError('subclasses must override _criterion()!')
        return 0.0

    def _to_tensor(self, data):
        raise NotImplementedError('subclasses must override _criterion()!')
        return 0.0

    def _train_epoch(self, train_loader, optimizer, epoch):
        """"""
        samples = {}
        metrics = {}
        ## switch to train mode
        self.net.train()
        self.mode = "train"
        loss_names = []
        for batch_idx, data in enumerate(train_loader):
            data_in, data_out = self._pack_data(data)

            ## forward
            ## metrics and loss
            data_pred = self.net(data_in, train_mode=True)
            losses = self._criterion(data_pred, data_out, self.cfg)
            loss = losses["loss_sum"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

#              ## check grads inf/nan
            #  for k,p in self.net.named_parameters():
            #      if p.grad is not None:
            #          if torch.sum(torch.isnan(p.grad)):
            #              print("Nan detected: ", k, p.shape)
            #          else:
            #              print(k, torch.max(p.grad), torch.min(p.grad))

            ## metrics
            metric = self._metric(data_pred, data_out)
            self._update_metrics(metrics, losses, metric)
            #  tf.logging.info("metric: %d"%(t-time.time()))
            #  t = time.time()

            ## collect samples
            if batch_idx+1 == len(train_loader):
                samples["sample_pred"] = data_pred
                samples["sample_input"] = data_in
                samples["sample_gnd"] = data_out

            # print statistics
            if batch_idx % self.cfg.OPTM.LOG_INTERVAL == 0:
                tf.logging.info('Train Epoch: {} [{}*({:.0f}%), {}]'.format(
                    epoch, len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    batch_idx
                ))
                tf.logging.info("Losses: " + ", ".join(["%s: %.4f"%(k,losses[k])
                                           for k in sorted(losses.keys())]))
            loss_names = losses.keys()
            del loss, data_pred, data_out, data_in, losses

        batch_idx += 1
        metrics = self._normalize_metrics(metrics, batch_idx)
        tf.logging.info("Train set: ")
        tf.logging.info("Losses: " + ", ".join([
            "%s: %.4f"%(k, metrics[k]) for k in sorted(loss_names)]))
        tf.logging.info('Metrics: ' + ", ".join([
            "%s: %.4f"%(k, metrics[k]) for k in sorted(metrics.keys())
            if k not in loss_names and isinstance(metrics[k], float)
        ]))

        for k in sorted(metrics.keys()):
            if isinstance(metrics[k], (list, dict)):
                tf.logging.info(
                    "Per class metric, " + "%s: %s"%(k, ",".join([str(v) for v in metrics[k]]))
                )
        return metrics, samples

    def _validate_epoch(self, val_loader):
        samples = {}
        metrics = {}
        # switch to evaluate mode
        self.net.eval()
        self.mode = "test"
        loss_names = []
        for batch_idx, data in enumerate(val_loader):
            #  data = self._to_tensor(data)
            data_in, data_out = self._pack_data(data)

            ## forward
            data_pred = self.net(data_in, train_mode=False)
            ## metrics and loss
            losses = self._criterion(data_pred, data_out, self.cfg)

            metric = self._metric(data_pred, data_out)
            self._update_metrics(metrics, losses, metric)

            ## collect samples
            if batch_idx == 1:
                samples["sample_pred"] = data_pred
                samples["sample_input"] = data_in
                samples["sample_gnd"] = data_out
            loss_names = losses.keys()
            del data, data_pred, data_out, data_in, losses

        batch_idx += 1
        metrics = self._normalize_metrics(metrics, batch_idx)
        tf.logging.info("Validation set: ")
        tf.logging.info("Losses: " + ", ".join([
            "%s: %.4f"%(k, metrics[k]) for k in sorted(loss_names)]))
        tf.logging.info('Metrics: ' + ", ".join([
            "%s: %.4f"%(k, metrics[k]) for k in sorted(metrics.keys())
            if k not in loss_names and isinstance(metrics[k], float)
        ]))

        for k in sorted(metrics.keys()):
            if isinstance(metrics[k], (list, dict)):
                tf.logging.info(
                    "Per class metric, " + "%s: %s"%(k, ",".join([str(v) for v in metrics[k]]))
                )

        return metrics, samples
