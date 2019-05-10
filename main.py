#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : main.py
# Author            : Hai-Yong Jiang <haiyong.jiang1990@hotmail.com>
# Date              : 19.08.2018
# Last Modified Date: 10.05.2019
# Last Modified By  : Hai-Yong Jiang <haiyong.jiang1990@hotmail.com>
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
import tensorflow as tf
from multiprocessing import cpu_count

from utils.config import cfg, cfg_setup, cfg_print, cfg_net_module, cfg_data_loader, cfg_trainer_module
from utils.train_callbacks import TensorboardLoggerCallback, ModelSaverCallback, TrainSaverCallback


def print_net_arch(net):
    tf.logging.info("Net architecture: ")
    for k,v in net.state_dict().items():
        tf.logging.info(k+", "+str(v.shape))

## main
def main(restore=False, pretrain_model_path="", layers="all"):
    # Hyperparameters
    cfg_setup()
    cfg_print()
    tf.logging.info("GPU usage:")
    epochs = cfg.OPTM.EPOCHS
    batch_size = cfg.OPTM.BATCH_SIZE

    # Training on x samples and validating on x samples
    threads = 2
    #  threads = 0
    torch.manual_seed(cfg.SEED)

    #### set up callbacks
    tb_logs_cb = TensorboardLoggerCallback(cfg)
    model_saver_cb = ModelSaverCallback(cfg)
    train_saver_cb = TrainSaverCallback(cfg)

    # Define our neural net architecture
    net = cfg_net_module(cfg.MODEL.NET)(cfg).cuda()
    print_net_arch(net)

    ## setup model
    predictor = cfg_trainer_module(cfg.MODEL.TRAINER)(cfg, net)
    if restore and os.path.exists(pretrain_model_path):
        tf.logging.info("Pretrained model: " + pretrain_model_path)
        predictor.restore_model(pretrain_model_path)

    ## load train data
    train_ds = cfg_data_loader(cfg.DATA.DATASET)("train", cfg)
    train_loader = DataLoader(train_ds, batch_size,
                              sampler=RandomSampler(train_ds),
                              #  sampler=SequentialSampler(train_ds),
                              num_workers=threads,
                              pin_memory=True)
    ## load test data
    valid_ds = cfg_data_loader(cfg.DATA.DATASET)("train", cfg)
    valid_loader = DataLoader(valid_ds, batch_size,
                              sampler=SequentialSampler(valid_ds),
                              num_workers=threads,
                              pin_memory=True)

    tf.logging.info("Training on {} samples and validating on {} samples "
          .format(len(train_loader.dataset), len(valid_loader.dataset)))
    predictor.train(train_loader, valid_loader, epochs,
                          callbacks=[tb_logs_cb, model_saver_cb, train_saver_cb],
                          layers=layers)
    tf.logging.info("Training is done.")


if __name__ == "__main__":
    main()

