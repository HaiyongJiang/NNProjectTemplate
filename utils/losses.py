#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : /home/haiyong/Project/semantic_urban/footprint_vae/srcs/nn/losses.py
# Author            : Hai-Yong Jiang <haiyong.jiang1990@hotmail.com>
# Date              : 22.08.2018
# Last Modified Date: 07.05.2019
# Last Modified By  : Hai-Yong Jiang <haiyong.jiang1990@hotmail.com>
import torch
import numpy as np
import sys
from nn.config import cfg


## **************************************************************************
## polygon utility functions
def _poly_area(vertices):
    """
    params:
        @vertices: [nbatch, nvts, 2]
    returns:
        @area: [nbatch], area for each polygon
    """
    nbatch, nvts, _ = vertices.size()
    idxs1 = np.arange(0, nvts)
    idxs2 = np.arange(0, nvts)
    idxs2[:nvts-1] = idxs2[1:]
    idxs2[-1] = 0

    area = torch.sum(vertices[:, idxs1, 0]*vertices[:, idxs2, 1]
            - vertices[:, idxs1, 1]*vertices[:, idxs2, 0], dim=1)
    return area

def _poly_arc(vertices):
    """
    params:
        @vertices: [nbatch, nvts, 2]
    returns:
        @arc: [nbatch], arc length for each polygon
    """
    nbatch, nvts, _ = vertices.size()
    idxs1 = np.arange(0, nvts)
    idxs2 = np.arange(0, nvts)
    idxs2[:nvts-1] = idxs2[1:]
    idxs2[-1] = 0

    arc = torch.norm(vertices[:, idxs2] - vertices[:, idxs1], dim=2)
    arc = torch.sum(arc, dim=1)
    return arc

def _poly_corner_angle(vertices):
    """
    params:
        @vertices: [nbatch, nvts, 2]
    returns:
        @cosangle: [nbatch], cosangle of each corner.
    """
    nbatch, nvts, _ = vertices.size()
    idxs1 = np.arange(0, nvts)
    idxs2 = np.arange(0, nvts)
    idxs2[:nvts-1] = idxs2[1:]
    idxs2[-1] = 0
    idxs3 = np.arange(0, nvts)
    idxs3[:nvts-2] = idxs2[2:]
    idxs3[-2] = 0
    idxs3[-1] = 1

    ## cosangles
    vec1 = vertices[:, idxs2] - vertices[:, idxs1]
    vec2 = vertices[:, idxs3] - vertices[:, idxs2]
    vec1 = vec1/(torch.norm(vec1, dim=2, keepdim=True)+1e-5)
    vec2 = vec2/(torch.norm(vec2, dim=2, keepdim=True)+1e-5)
    cosangle = torch.sum(vec1*vec2, dim=2)
    return cosangle


## **************************************************************************
## loss utility functions
def loss_polyarea(poly_pred, poly_gnd, keep_batch=False):
    areadiff = (_poly_area(poly_pred) - _poly_area(poly_gnd))**2
    if keep_batch:
        loss = areadiff
    else:
        loss = areadiff.mean()
    return loss

def loss_polyarc(poly_pred, poly_gnd, keep_batch=False):
    arcdiff = (_poly_arc(poly_pred) - _poly_arc(poly_gnd))**2
    if keep_batch:
        loss = arcdiff
    else:
        loss = arcdiff.mean()
    return loss


def loss_regularization(model, name="l1"):
    reg_loss = 0
    if name == "l1":
        for param in model.parameters():
            reg_loss += torch.norm(param, 1)
    elif name == "l2":
        for param in model.parameters():
            reg_loss += torch.norm(param, 2)
    return reg_loss


def loss_vts(pred_poly, target_poly, keep_batch=False):
    """
    Do not know why. The chamfer distance and the torch.cat will lead to memory leak.
    If torch.zeros is used, no problem will be reported.
    params:
        @pred_mesh: [nbatch, nvts, 2]
        @targ_mesh: [nbatch, nvts, 2]
    returns:
        loss
    """
    nbatch, nvts, _ = pred_poly.size()
    dist1 = torch.sum(((pred_poly - target_poly))**2, dim=2)
    if keep_batch:
        loss = torch.mean(dist1, dim=1)
    else:
        loss = torch.mean(dist1)

    return loss

def loss_vts_w_label(pred_poly, target_poly, keep_batch=False):
    """
    Do not know why. The chamfer distance and the torch.cat will lead to memory leak.
    If torch.zeros is used, no problem will be reported.
    params:
        @pred_mesh: [nbatch, nvts, 3], (x,y,l)
        @targ_mesh: [nbatch, nvts, 3]
    returns:
        loss
    """
    nbatch, nvts, _ = pred_poly.size()
    label = target_poly[:, :, -1:].contiguous().expand(nbatch, nvts, 2).ge(0)
    pred_poly = torch.masked_select(pred_poly[:, :, :2], label)
    target_poly = torch.masked_select(target_poly[:, :, :2], label)
    return torch.mean(((pred_poly - target_poly))**2)

def loss_label(pred_poly, target_poly, keep_batch=False):
    """
    Do not know why. The chamfer distance and the torch.cat will lead to memory leak.
    If torch.zeros is used, no problem will be reported.
    params:
        @pred_mesh: [nbatch, nvts, 1], (x,y,l)
        @targ_mesh: [nbatch, nvts, 1]
    returns:
        loss
    """
    nbatch, nvts, _ = pred_poly.size()
    targ_label = target_poly[:, :, -1:].contiguous()
    pred_label = pred_poly[:, :, -1:].contiguous()
    loss = torch.nn.functional.binary_cross_entropy(pred_label, targ_label)
    return torch.mean(loss)

def get_corners(poly):
    """
    params:
        @poly: a polygon [nbatch, nvts, 2]
    returns:
        @corners: corner angle representation of each corner
    """
    npoly = poly.size()[1]
    idxs = torch.tensor(list(range(1,npoly)) + [0,], dtype=torch.long).cuda()
    poly_n = torch.index_select(poly, 1, idxs)
    poly = (poly - poly_n)/ (torch.norm(poly-poly_n, dim=-1, keepdim=True) + 1e-8)
    return poly

def loss_corner(pred_poly, target_poly, keep_batch=False):
    """
    Do not know why. The chamfer distance and the torch.cat will lead to memory leak.
    If torch.zeros is used, no problem will be reported.
    params:
        @pred_mesh: [nbatch, nvts, 2]
        @targ_mesh: [nbatch, nvts, 2]
    returns:
        loss
    """
    nbatch, nvts, _ = pred_poly.size()
    pred_corner = get_corners(pred_poly)
    target_corner = get_corners(target_poly)
    dist1 = torch.sum(((pred_corner- target_corner))**2, dim=2)
    if keep_batch:
        loss = torch.mean(dist1, dim=1)
    else:
        loss = torch.mean(dist1)
    return loss

def loss_vts_loop(pred_poly, target_poly):
    """
    Do not know why. The chamfer distance and the torch.cat will lead to memory leak.
    If torch.zeros is used, no problem will be reported.
    params:
        @pred_mesh: [nbatch, nvts, 2]
        @targ_mesh: [nbatch, nvts, 2]
    returns:
        loss
    """
    nbatch, nvts, _ = pred_poly.size()

    idxs = torch.tensor(list(range(1, nvts)) + [0,], dtype=torch.long).cuda()
    dist1 = torch.sum(((pred_poly - target_poly))**2, dim=2)
    loss = torch.mean(dist1)
    #  print("initial: ", loss.item())
    for ii in range(1, nvts):
        pred_poly = torch.index_select(pred_poly, 1, idxs)
        dist = torch.sum(((pred_poly - target_poly))**2, dim=2)
        loss = torch.min(torch.mean(dist), loss)
    #  print("afterwards: ", loss.item())

    return loss


def loss_edges(pred_poly, target_poly, keep_batch=False):
    """
    Do not know why. The chamfer distance and the torch.cat will lead to memory leak.
    If torch.zeros is used, no problem will be reported.
    params:
        @pred_mesh: [nbatch, nvts, 2]
        @targ_mesh: [nbatch, nvts, 2]
    returns:
        loss
    """
    nbatch, nvts, _ = pred_poly.size()

    ## interpolation for dense point distance
    idxs1 = np.arange(0, nvts)
    idxs2 = np.arange(0, nvts)
    idxs2[:nvts-1] = idxs2[1:]
    idxs2[-1] = 0
    pred_edges  = pred_poly[:, idxs1] - pred_poly[:, idxs2]
    target_edges  = target_poly[:, idxs1] - target_poly[:, idxs2]

    pred_edges = torch.norm(pred_edges, dim=2, keepdim=True)
    target_edges = torch.norm(target_edges, dim=2, keepdim=True)
    unit = torch.mean(target_edges)
    pred_edges = pred_edges/unit
    target_edges = target_edges/unit

    dist1 = torch.sum((pred_edges- target_edges)**2, dim=2)
    if keep_batch:
        loss = torch.mean(dist1, dim=1)
    else:
        loss = torch.mean(dist1)

    return loss



def loss_corner_reg(pred_poly, target_poly):
    """
    corner regularity, requiring a corner to be flat or orthgonal.
    params:
        @pred_poly: [nbatch, nvts, 2]
        @target_poly: [nbatch, nvts, 2]
    returns:
        loss
    """
    nbatch, nvts, _ = pred_poly.size()
    pred_cosangles = _poly_corner_angle(pred_poly)
    targ_cosangles = _poly_corner_angle(target_poly)

    #  loss = torch.mean(((pred_cosangles-targ_cosangles))**2)
    #  loss = torch.mean((torch.ge(1-targ_cosangles, 0.05)\
    #                     .to("cuda", dtype=torch.float32)*(pred_cosangles-targ_cosangles))**2)

    eps = 0.00
    #  loss1 = torch.mean((torch.ge(targ_cosangles, 1.0-eps)\
    #                     .to("cuda", dtype=torch.float32)*(pred_cosangles-targ_cosangles))**2)
    #  loss2 = torch.mean((torch.le(torch.abs(targ_cosangles), eps)\
    #                     .to("cuda", dtype=torch.float32)*(pred_cosangles-targ_cosangles))**2)

    #  loss1 = (torch.ge(targ_cosangles, 1.0-eps)\
    #                     .to("cuda", dtype=torch.float32)*torch.abs(pred_cosangles-1.0))
    #  loss2 = (torch.le(torch.abs(targ_cosangles), eps)\
    #                     .to("cuda", dtype=torch.float32)*torch.abs(pred_cosangles))

    loss1 = torch.clamp((pred_cosangles-1.0)**2-eps, min=0.0)
    loss2 = torch.clamp((pred_cosangles)**2-eps, min=0.0)

    loss = torch.min(loss1, loss2).mean()
    #  loss = torch.mean(((pred_cosangles-1.0))**2)
    return loss


def loss_ssim(pred_poly, targ_poly):
    """ ssim loss, refer to wiki
    """
    mu1 = torch.mean(pred_poly, dim=1)
    mu2 = torch.mean(targ_poly, dim=1)
    sigma1 = torch.mean(pred_poly*pred_poly, dim=1) - mu1*mu1
    sigma2 = torch.mean(targ_poly*targ_poly, dim=1) - mu2*mu2
    sigma12 = torch.mean(pred_poly*targ_poly, dim=1) - mu1*mu2
    c = 0.01**2

    loss_ssim = ((2*mu1*mu2 + c)*(2*sigma12+c))/((mu1*mu1+mu2*mu2+c)*(sigma1 + sigma2 + c))
    return (1-loss_ssim.mean())/2.0

