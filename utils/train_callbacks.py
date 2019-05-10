#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : utils/train_callbacks.py
# Author            : Hai-Yong Jiang <haiyong.jiang1990@hotmail.com>
# Date              : 19.08.2018
# Last Modified Date: 10.05.2019
# Last Modified By  : Hai-Yong Jiang <haiyong.jiang1990@hotmail.com>


#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : nn/train_callbacks.py
# Author            : Hai-Yong Jiang <haiyong.jiang1990@hotmail.com>
# Date              : 15.05.2018
# Last Modified Date: 11.08.2018
# Last Modified By  : Hai-Yong Jiang <haiyong.jiang1990@hotmail.com>
import os
import torch
import tensorflow as tf
from tensorboardX import SummaryWriter

class Callback:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def add_dict(self, writer, val_dict, prefix, epoch_id, val_type="scalar"):
        for k,v in val_dict.items():
            if val_type=="scalar":
                writer.add_scalar(prefix+"/"+k, v, epoch_id)
            elif val_type=="hist":
                writer.add_histogram(prefix+"/"+k, v, epoch_id)


class TensorboardLoggerCallback(Callback):
    def __init__(self, cfg):
        """
            Callback intended to be executed at each epoch
            of the training which goal is to add valuable
            information to the tensorboard logs such as the losses
            and accuracies
        Args:
            path_to_files (str): The path where to store the log files
        """
        self.path_to_files = cfg.DATA.RES_LOG_PATH
        self.path_to_model = cfg.DATA.RES_MODEL_PATH
        self.save_best = cfg.OPTM.SAVE_BEST
        self.loss = 1e10

    def __call__(self, *args, **kwargs):
        if kwargs['step_name'] != "epoch":
            return

        epoch_id = kwargs['epoch_id']

        ## add loss for tensorboard visualization
        self.writer = SummaryWriter(self.path_to_files)
        for name in ["train_loss", "val_loss"]:
            self.add_dict(self.writer, kwargs[name],
                            'data/%s'%(name), epoch_id)
        self.add_dict(self.writer, {"lr": kwargs['lr']}, 'data/learning_rate', epoch_id)
        lr = kwargs['lr']

        if "train_sample" in kwargs and epoch_id>0:
            sample = kwargs["train_sample"]
            for k in sample:
                if k.endswith("weight"):
                    if "param_" in k:
                        self.add_dict(self.writer, {k:lr*sample[k]}, 'param/', epoch_id, "hist")
                    if "update_" in k:
                        self.add_dict(self.writer, {k:lr*sample[k]}, 'update/', epoch_id, "hist")

        self.writer.close()


def saver_keep_last_k(filename_ref, k):
    folder = os.path.dirname(filename_ref)
    filename = os.path.basename(filename_ref)
    basename, ext = os.path.splitext(filename)
    filelist = []
    for f in os.listdir(folder):
        if f.startswith(basename) and f.endswith(ext) and f!=filename:
            filelist.append(f)
    filelist = sorted(filelist)[:-k]
    for f in filelist:
        f = folder + "/" + f
        if os.path.exists(f):
            os.remove(f)
        else:
            print("Path does not exist: " + f)


class ModelSaverCallback(Callback):
    def __init__(self, cfg, save_best=False, verbose=False):
        """
            Callback intended to be executed each time a whole train pass
            get finished. This callback saves the model in the given path
        Args:
            verbose (bool): True or False to make the callback verbose
            path_to_model (str): The path where to store the model
        """
        self.verbose = verbose
        self.path_to_model = cfg.DATA.RES_MODEL_PATH
        self.save_best = save_best
        self.suffix = ""

    def set_suffix(self, suffix):
        """

        Args:
            suffix (str): The suffix to append to the model file name
        """
        self.suffix = suffix

    def __call__(self, *args, **kwargs):
        if kwargs['step_name'] not in ["train", "epoch"]:
            return

        pth = self.path_to_model + self.suffix
        net = kwargs['net']
        torch.save(net.state_dict(), pth)

        if self.verbose:
            tf.logging.info("Model saved in {}".format(pth))


class TrainSaverCallback:
    def __init__(self, cfg, max_save=5):
        self.log_interval = cfg.OPTM.IMG_LOG_INTERVAL
        self.cfg = cfg
        self.root_path = cfg.DATA.RES_IMG_PATH

    def __call__(self, *args, **kwargs):
        """ Save Input/Target/Predict/Diff, Corner/Line/Poly_maps """
        if kwargs['step_name'] != "epoch":
            return
        epoch = 0
        if 'epoch_id' in kwargs:
            epoch = kwargs['epoch_id']

        if epoch%self.log_interval!= 0:
            return

        ## save the meshes
        for split in ["train", "val"]:
            if len(kwargs[split + "_sample"]) == 0:
                continue
            sample = kwargs[split + "_sample"]

            ## TODO: custom your data logger

