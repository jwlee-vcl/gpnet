from __future__ import division	
import sys
import os
from datetime import date
import platform
import time

import numpy as np
import numpy.linalg as LA
import scipy.linalg as SLA

import os.path
import cv2 as cv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.utils as vutils
import torchvision.models as tmodels

import math
import models.models_gpnet as models_gpnet
from .base_model import BaseModel
from . import networks_gpnet

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from models import vis_utils
from models import hl_utils
from models import gpnet_utils

EPSILON = 1e-8

def cvPoint(pt):
    return (int(pt[0]), int(pt[1]))

def compute_hl_np(hl, sz, eps=1e-6):
    (a,b,c) = hl
    if b < 0:
        a, b, c = -a, -b, -c
    b = np.maximum(b, eps)
    
    left = np.array([-1.0, (a - c)/b])        
    right = np.array([1.0, (-a - c)/b])

    # scale back to original image    
    scale = sz[1]/2    
    left = scale*left
    right = scale*right
    
    return [np.squeeze(left), np.squeeze(right)]

def cos_dis(l,p):
    l = np.reshape(l, (-1,3))
    p = np.reshape(p, (3,-1))

    l = l/LA.norm(l, axis=1, keepdims=True)
    p = p/LA.norm(p, axis=1, keepdims=True)

    return np.abs(np.dot(l,p))


class GPNet(BaseModel):
    def name(self):
        return 'GPNet'

    def __init__(self, opt, _isTrain):
        BaseModel.initialize(self, opt)

        self.opt = opt
        self.mode = opt.mode
        self.num_input = 3 #opt.input_nc

        if self.mode == 'ResNet':
            new_model = models_gpnet.GPNet(opt)
        else:
            print('ONLY SUPPORT Ours_Bilinear')
            sys.exit()
        
        model_name = '_best_{}_gpnet_mode_ResNet_lr_0.0004'.format(opt.dataset)            

        if not _isTrain:            
            model = self.load_network(new_model, 'G', model_name)
            new_model.load_state_dict(model)
        else:   
            model = self.load_network(new_model, 'G', model_name)
            new_model.load_state_dict(model, strict=False)
            

        new_model = torch.nn.parallel.DataParallel(new_model.cuda(), device_ids = [0])
        self.netG = new_model
        self.criterion = networks_gpnet.GPNetLoss()      
        if not _isTrain:            
            self.vis_dir = '../2020_gpnet/GPNet_{}_{}/'.format(opt.dataset, date.today())
            os.makedirs(self.vis_dir, exist_ok=True)

        if _isTrain:      
            self.old_lr = opt.lr
            self.netG.train()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(0.9, 0.999), weight_decay=1e-4)
            self.scheduler = networks_gpnet.get_scheduler(self.optimizer_G, opt)
        else:
            os.makedirs(self.vis_dir, exist_ok=True)

        print('---------- Networks initialized -------------')
        networks_gpnet.print_network(self.netG)
        print('-----------------------------------------------')

    def set_writer(self, writer):
        self.writer = writer

    def set_input(self, stack_imgs, stack_segs, stack_seg_map, stack_seg_mask, targets):
        self.input_x = stack_imgs
        self.input_l = stack_segs
        self.input_s = stack_seg_map
        self.input_m = stack_seg_mask
        self.targets = targets

    def forward(self):
        self.input_images = self.input_x.cuda()
        self.input_segs = self.input_l.cuda()
        self.input_seg_map = self.input_s.cuda()
        self.input_seg_mask = self.input_m.cuda()

        self.pred_dict = self.netG.forward(self.input_images, self.input_segs, 
                                           self.input_seg_map, self.input_seg_mask, self.targets)

    def write_summary_train(self, mode_name, input_images, input_segs,
                            zvp_term, hl_term,
                            targets, n_iter):

        num_write = 6

        self.writer.add_scalar(mode_name + '/zvp_term', zvp_term, n_iter)
        self.writer.add_scalar(mode_name + '/hl_term', hl_term, n_iter)				

        self.writer.add_image(mode_name + '/img', vutils.make_grid(input_images.data[:num_write,:,:,:].cpu(), normalize=True), n_iter)


    def write_summary_val(self, mode_name, input_images, input_segs,
                          pred_dict, targets, n_iter):

        num_write = 6

        self.writer.add_image(mode_name + '/img', vutils.make_grid(input_images.data[:num_write,:,:,:].cpu(), normalize=True), n_iter)

    def backward_G(self, n_iter):
        # Combined loss
        self.loss, loss_dict = self.criterion(self.pred_dict, self.targets)

        if n_iter%20 == 0:                                               
            print("Train loss is %f "%self.loss)            
            print(['{} {:.05f}'.format(key, loss_dict[key]) for key in loss_dict.keys()])            

        self.loss_var = self.criterion.get_loss_var()
        self.loss_var.backward()

    def optimize_parameters(self, n_iter):
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward_G(n_iter)
        self.optimizer_G.step()

    def evaluate_horizon(self, input_x, input_l, input_s, input_m, targets, n_iter, write_summary):
        # switch to evaluation mode
        with torch.no_grad():			
            input_imgs = input_x.cuda()
            input_segs = input_l.cuda()
            input_seg_map = input_s.cuda()
            input_seg_mask = input_m.cuda()
            
            pred_dict = self.netG.forward(input_imgs, input_segs, input_seg_map, input_seg_mask, targets)
                        
            horizon_error = self.criterion.compute_horizon_error(pred_dict, targets)

            if write_summary:
                print('==================== WRITING EVAL SUMMARY ==================')
                self.write_summary_val('Eval', input_imgs, input_segs,
                                        pred_dict, targets, n_iter)

            return horizon_error

    def test(self, input_x, input_l, input_s, input_m, targets):
        with torch.no_grad():           
            tic = time.time()
            input_imgs = input_x.cuda()
            input_segs = input_l.cuda()
            input_seg_map = input_s.cuda()
            input_seg_mask = input_m.cuda()
            toc_uploading = time.time() - tic
            
            pred_dict = self.netG.forward(input_imgs, input_segs, input_seg_map, input_seg_mask, in_dict=targets)                        
        return pred_dict

    def test_angle_horizon_error(self, input_x, input_l, input_s, input_m, targets):
        with torch.no_grad():			
            input_imgs = input_x.cuda()
            input_segs = input_l.cuda()
            input_seg_map = input_s.cuda()
            input_seg_mask = input_m.cuda()

            pred_dict = self.netG.forward(input_imgs, input_segs, input_seg_map, input_seg_mask, in_dict=targets)
                        
            rotation_error, roll_error, pitch_error, fov_error = self.criterion.compute_angle_error(
                pred_dict, targets, stack_error=True)

            horizon_error = self.criterion.compute_horizon_error(
                pred_dict, targets, stack_error=True)

        return rotation_error, roll_error, pitch_error, fov_error, horizon_error
    
    def switch_to_train(self):
        self.netG.train()

    def switch_to_eval(self):
        self.netG.eval()

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)        

    def update_learning_rate(self):
        self.scheduler.step()		
        lr = self.optimizer_G.param_groups[0]['lr']
        print('Current learning rate = %.4f' % lr)






