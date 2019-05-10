#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : nn/layers.py
# Author            : Hai-Yong Jiang <haiyong.jiang1990@hotmail.com>
# Date              : 29.04.2019
# Last Modified Date: 30.04.2019
# Last Modified By  : Hai-Yong Jiang <haiyong.jiang1990@hotmail.com>
import math
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn import init


class GCNLayer(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True, relu=nn.ReLU(), use_bn=True):
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_f0 = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight_f1 = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.relu = relu
        if use_bn:
            self.bn = nn.BatchNorm1d(out_features)
        else:
            self.bn = lambda x: x


    def reset_parameters(self):
        init.kaiming_uniform_(self.weight_f0, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_f1, a=math.sqrt(5))
        if self.bias is not None:
            bound = 0.0001
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input, adj):
        """
        inputs:
            @input: [B, N, Fin]
        """
        f0 = torch.matmul(input, self.weight_f0) #[B, N, Fout]
        f1 = torch.matmul(input, self.weight_f1) #[B, N, Fout]
        output = torch.matmul(f1.transpose(2,1), adj).transpose(2,1).contiguous() + f0 #[B, N, Fout]
        if self.bias is not None:
            output = output + self.bias
        output = self.relu(output)
        output = self.bn(output.transpose(2,1).contiguous()).transpose(2,1).contiguous() #[B, N, Fout]
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



