#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : experiments/exp_example.py
# Author            : Hai-Yong Jiang <haiyong.jiang1990@hotmail.com>
# Date              : 23.11.2018
# Last Modified Date: 10.05.2019
# Last Modified By  : Hai-Yong Jiang <haiyong.jiang1990@hotmail.com>
import os, sys
import time
import argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../")
from utils.config import cfg, cfg_last_timestamp
from main import main

cfg.GPU_IDS = '3'
cfg.DATA.DATASET = "none"
cfg.MODEL.NET = "none"
cfg.MODEL.TRAINER = ""
cfg.MODEL.TIME_STAMP = time.strftime("%Y%m%d%H%M", time.localtime())
cfg.MODEL.COMMENTS = "basic"

## optimizing config
cfg.OPTM.LR = 5e-4
cfg.OPTM.EPOCHS = 500
cfg.OPTM.BATCH_SIZE = 32

cfg.LOG.PARAMS = False
cfg.LOSS.WEIGHTS = {"loss_vts": 3.0, "loss_reg": 0.0}
cfg.MODEL.COMMENTS +=  "_" +  "_".join([k.replace("loss_", "") for k in cfg.LOSS.WEIGHTS if cfg.LOSS.WEIGHTS[k]>0.0])


## Training
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU_IDS
print("Train on Gpu " + str(cfg.GPU_IDS))

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="train", help="train/test")
parser.add_argument("--restore", action='store_true', default=False,
        help='restore from last saving')
#  parser.add_argument("--modelpath", type=str, default="",
#          help='the model file path')
parser.add_argument("--stamp", type=str, default="", help="timestamp used for restoring")
parser.add_argument("--data_split", type=str, default="test", help="train/test/val")
parser.add_argument("--no_cuda", action='store_true', default=False,
        help='if use cuda or not')

## arguments
args = parser.parse_args()
if args.restore:
    cfg.MODEL.TIME_STAMP = cfg_last_timestamp()
if args.stamp:
    cfg.MODEL.TIME_STAMP = args.stamp
if args.no_cuda:
    cfg.USE_CUDA = not args.no_cuda
if args.mode == "train":
    ## train the network with param loss
    main()



