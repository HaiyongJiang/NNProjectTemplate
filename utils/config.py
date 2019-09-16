#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : utils/config.py
# Author            : Hai-Yong Jiang <haiyong.jiang1990@hotmail.com>
# Date              : 19.08.2018
# Last Modified Date: 16.09.2019
# Last Modified By  : Hai-Yong Jiang <haiyong.jiang1990@hotmail.com>
## **************************************************************************
from easydict import EasyDict as edict
import logging
import os
## PLEASE KEEP THE IMPORTING ORDER OF torch and tensorflow, otherwise memory corrpution will be reported.
import torch
import tensorflow as tf
import numpy as np

## basic configurations
cfg = edict()
cfg.DATA = edict()
cfg.MODEL = edict()
cfg.OPTM = edict()
cfg.LOSS = edict()
cfg.LOG = edict()

cfg.DATA.RES_LOG_PATH = "../outputs/logs/"
cfg.DATA.ROOT_PATH = "../data/"
cfg.DATA.NSAMPLE_TRAIN = 10240
cfg.DATA.NSAMPLE_VAL = 1024
cfg.DATA.NSAMPLE_TEST = 512
cfg.DATA.DATASET = "default"

cfg.MODEL.IDENTIFIER = 'xx'
cfg.MODEL.TF_LOG_SETTING = 'xx'
cfg.MODEL.TIME_STAMP = ""
cfg.MODEL.COMMENTS = ""
cfg.MODEL.NET = "default"
cfg.MODEL.TRAINER = "Predictor"
cfg.MODEL.iEPOCH = 0
cfg.MODEL.BN = "BN" ##[BN, INST_BN, RE_BN]
cfg.MODEL.BN_DECAY = 0.1
cfg.MODEL.USE_BN_DECAY = False
cfg.MODEL.VAL_EPOECH_FREQ = 1

cfg.GPU_IDS = '0'
cfg.VERBOSE = False
cfg.SEED = 100

cfg.OPTM.LR = 0.005
## [xx| "MultiStep"]
cfg.OPTM.LR_SCHEDULER = "ReduceLROnPlateau"
#### {"milestones": [2000, 3500, 4500, 5000], "gamma":0.1} for multiple steps
cfg.OPTM.LR_SCHEDULER_PARAM = {"factor": 0.3, "patience":8,
                               "cooldown":0, "min_lr": 1e-8}
cfg.OPTM.OPTIMIZER = "ADAM"
cfg.OPTM.BATCH_SIZE = 32
cfg.OPTM.EPOCHS = 500
cfg.OPTM.EARLY_STOPPING = True
cfg.OPTM.SAVE_BEST = True
cfg.OPTM.FINETUNE = False
## if logging parameters or not.
cfg.OPTM.LOG_PARAMS = False
cfg.OPTM.LOG_INTERVAL = 50
cfg.OPTM.IMG_LOG_INTERVAL = 5

## scale of vertices for easy optimization
cfg.LOSS.LOSS_NAME = "mse"
cfg.LOSS.REGULARIZATION = "l1"
cfg.LOSS.WEIGHTS = {}

def cfg_setup():
    ## checking
    cfg.MODEL.IDENTIFIER = "_".join(
        [cfg.DATA.DATASET, cfg.MODEL.NET, cfg.MODEL.TIME_STAMP,
         cfg.MODEL.COMMENTS, "lr%f"%(cfg.OPTM.LR)])
    identifier = cfg.MODEL.IDENTIFIER

    FILE_PATH = "../outputs/results/" + identifier + "/"
    cfg.MODEL.FILE_TF_LOG = FILE_PATH + "tf_logs.txt"
    cfg.DATA.RES_LOG_PATH = FILE_PATH + "/tf_logs/"
    cfg.DATA.RES_MODEL_PATH = FILE_PATH  + cfg.DATA.DATASET + "_" + cfg.MODEL.NET + ".pth"
    cfg.DATA.RES_IMG_PATH = FILE_PATH  + "/images/"

    for path in [
        os.path.dirname(cfg.DATA.RES_LOG_PATH),
        os.path.dirname(cfg.DATA.RES_IMG_PATH),
        ]:
        if not os.path.exists(path):
            os.makedirs(path)

    if 'TF_LOG_SETTING' not in locals() and \
            'TF_LOG_SETTING' not in globals():
        FORMAT = '%(asctime)s %(levelname)-8s %(message)s'
        logging.basicConfig(format=FORMAT)
        log = logging.getLogger('tensorflow')
        if cfg.VERBOSE:
            log.setLevel(logging.DEBUG)
        else:
            log.setLevel(logging.INFO)
        fh = logging.FileHandler( cfg.MODEL.FILE_TF_LOG )
        fh.setLevel(logging.INFO)
        log.addHandler(fh)
        log.propagate = False
    TF_LOG_SETTING = 1


def cfg_print():
    tf.logging.info("Start training with the following configs: ")
    tf.logging.info("MODEL: " + cfg.MODEL.NET)
    tf.logging.info(cfg.MODEL)
    tf.logging.info("IDENTIFIER: " + cfg.MODEL.IDENTIFIER)
    tf.logging.info("COMMENTS: " + cfg.MODEL.COMMENTS)

    tf.logging.info("Data:" + cfg.DATA.DATASET)
    tf.logging.info(cfg.DATA)

    tf.logging.info("OPTM: ")
    tf.logging.info(cfg.OPTM)

    tf.logging.info("LOSS: ")
    tf.logging.info(cfg.LOSS)


def cfg_last_timestamp():
    model_path = os.path.dirname(cfg.DATA.RES_MODEL_PATH)
    fname_last = ""
    for fname in sorted(os.listdir(model_path)):
        if fname.startswith(cfg.DATA.DATASET + "_" + cfg.MODEL.NET) \
                and os.path.exists(model_path + "/" + fname) \
                and cfg.MODEL.COMMENTS in fname:
            fname_last = fname
    for field in fname_last.split("_"):
        if len(field)>=12 and unicode(field, 'utf-8').isdecimal():
            break
    return field

def cfg_net_module(name):
    net_dict = {}
    return net_dict[name]

def cfg_data_loader(name):
    dataset_dict = {}
    from nn.loader import BP4DDb
    dataset_dict["BP4D"] = BP4DDb

    return dataset_dict[name]

def cfg_trainer_module(name):
    from nn.predictor import Predictor
    model_dict = {"predictor": Predictor,
                  #  "aupredictor": AUPredictor
                  }
    return model_dict[name]

