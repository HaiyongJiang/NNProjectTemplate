#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : nn/model.py
# Author            : Hai-Yong Jiang <haiyong.jiang1990@hotmail.com>
# Date              : 22.11.2018
# Last Modified Date: 10.05.2019
# Last Modified By  : Hai-Yong Jiang <haiyong.jiang1990@hotmail.com>
import torch
from torch import nn
#  from nn.resnet import resnet18, resnet34
import numpy as np
from utils.layers import GCNLayer
from utils import net_blocks


