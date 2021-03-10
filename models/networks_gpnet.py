from __future__ import division
import os
import sys
import math

import functools

import numpy as np
import numpy.linalg as LA

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse
from torch.autograd import Function
from torch.optim import lr_scheduler

from models import hl_utils

###############################################################################
# Functions
###############################################################################
VERSION = 4
EPSILON = 1e-8

def cosine_distance(v1, v2, dim=-1, keepdim=False):
    v1 = F.normalize(v1, p=2, dim=dim)
    v2 = F.normalize(v2, p=2, dim=dim)    
    return (v1*v2).sum(dim, keepdim).abs()

def normalize_lines_np(lines, axis=-1, eps=1e-6):
    (ab,_) = np.split(lines, [2], axis=axis)
    de = LA.norm(ab, axis=axis, keepdims=True)
    de = np.maximum(de, eps)
    return lines/de

def normalize_lines(line, dim=-1, eps=1e-6):
    (ab,_) = torch.split(line, [2,1], dim=dim)
    denorm = torch.norm(ab, dim=dim, keepdim=True)
    denorm = torch.max(denorm, torch.tensor(eps, device=denorm.device))[0]
    return line/denorm

def get_scheduler(optimizer, opt):
	if opt.lr_policy == 'lambda':
		def lambda_rule(epoch):
			lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
			return lr_l
		scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
	elif opt.lr_policy == 'step':
		scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_epoch, gamma=0.5)
	elif opt.lr_policy == 'plateau':
		scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
	else:
		return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
	return scheduler

def get_norm_layer(norm_type='instance'):
	if norm_type == 'batch':
		norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
	elif norm_type == 'instance':
		norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
	elif norm_type == 'none':
		norm_layer = None
	else:
		raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
	return norm_layer

def print_network(net_):
	num_params = 0
	for param in net_.parameters():
		num_params += param.numel()
	#print(net_)
	#print('Total number of parameters: %d' % num_params)

def rotm2eul_ypr(R):    
    cx = np.sqrt(R[2,0]*R[2,0] + R[2,2]*R[2,2])     
    x = np.arctan2( R[2,1], cx)
    y = np.arctan2(-R[2,0], R[2,2])
    z = np.arctan2(-R[0,1], R[1,1])
    return (x, y, z)

def eul2rotm_ypr(euler):
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(euler[0]),-np.sin(euler[0])],
                    [0, np.sin(euler[0]), np.cos(euler[0])]], dtype=np.float32)
  
    R_y = np.array([[ np.cos(euler[1]), 0, np.sin(euler[1])],
                    [0, 1, 0 ],
                    [-np.sin(euler[1]), 0, np.cos(euler[1])]], dtype=np.float32)
  
    R_z = np.array([[np.cos(euler[2]),-np.sin(euler[2]), 0],
                    [np.sin(euler[2]), np.cos(euler[2]), 0],
                    [0, 0, 1]], dtype=np.float32)                   
    return np.dot(R_z, np.dot(R_x, R_y))

def decompose_up_vector(v):
    x,y,z = torch.unbind(v, dim=-1)
    pitch = torch.asin(z)
    roll = torch.atan(-x/y)
    return pitch, roll

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

def compute_hl_error_np(gt_hl, est_hl, sz):
    '''
    gt_hl: a groud truth horizon line (left, right) 
    est_hl: an estimated horizon line (left, right)
    sz: a size of an image, (height, width)
    '''    
    err = np.maximum(np.abs(gt_hl[0][1] - est_hl[0][1]), # left side
                     np.abs(gt_hl[1][1] - est_hl[1][1])) # right side
    return err/sz[0]

def compute_hl(hl, dim=-1, eps=1e-6):        
    (a,b,c) = torch.split(hl, 1, dim=dim) # [b,3]
    hl = torch.where(b < 0.0, hl.neg(), hl)
    (a,b,c) = torch.split(hl, 1, dim=dim) # [b,3]
    b = torch.max(b, torch.tensor(eps, device=b.device))

    # compute horizon line
    left  = (a - c)/b  # [-1.0, ( a - c)/b]
    right = (-a - c)/b # [ 1.0, (-a - c)/b]

    return left, right

def _outer3(x):    
    return torch.matmul(
        torch.reshape(x, (-1, 3, 1)),
        torch.reshape(x, (-1, 1, 3))).view((-1, 9))

def structure_tensor_loss(pred, label, weight):
    '''
    pred: (N, 3)
    label: (N, 3)
    '''
    pred = _outer3(pred)
    label = _outer3(label)
    diff = (pred - label).abs()
    loss = diff.sum(dim=1)*weight
    return loss.mean()

# def compute_cands_score(pred_dict, targets, alpha=0.0, sigma=0.5):
#     gt_hl = targets['gt_hl'].cuda() # [b,3]
#     #seg_mask = targets['seg_mask'].cuda()
#     vseg_mask = targets['vseg_mask'].cuda()

#     pred_frames = pred_dict['frames'] # [b,n,9]
#     pred_focals = pred_dict['focals'] # [b,n,1]
            
#     act_map = pred_dict['act_maps'] # [b,n,h,w,3]

#     with torch.no_grad():
#         gt_hl = gt_hl.unsqueeze(1)
#         gt_l, gt_r = compute_hl(gt_hl, dim=-1) # [b,1,3]

#         pred_hls = pred_frames[:,:,3:6].clone()
#         pred_hls[:,:,2:3] = pred_focals*pred_hls[:,:,2:3]
        
#         est_ls, est_rs = compute_hl(pred_hls, dim=-1) # [b,n,3]

#         err_ls = torch.abs(est_ls - gt_l)
#         err_rs = torch.abs(est_rs - gt_r)

#         hl_errors = torch.max(err_ls, err_rs) # 0 ~ inf
#         hl_scores = hl_errors.neg().exp()

#         act_map = act_map[:,:,:,:,1:2] # [b,n,h,w,1]
#         vseg_mask = vseg_mask.unsqueeze(dim=1) # [b,n,h,w,1]
#         numer = (act_map*vseg_mask).sum(dim=2).sum(dim=2)
#         denorm = vseg_mask.sum(dim=2).sum(dim=2)
#         denorm = torch.max(denorm, torch.ones_like(denorm))
#         act_scores = numer/denorm # (bad) 0 ~ 1 (good)
        
#         scores = (hl_scores + act_scores)/2.0
#         scores = (-(((scores - 1.0)/sigma).pow(2)/2.0)).exp()    

#         return scores

def compute_cands_score(pred_dict, targets, alpha=0.0, sigma=0.5):
    gt_hl = targets['gt_hl'].cuda() # [b,3]
    gt_up_vector = targets['gt_up_vector'].cuda() # [b,3]
    gt_up_vector = gt_up_vector.unsqueeze(dim=1) # [b,1,3]
    
    pred_frames = pred_dict['frames'] # [b,n,9]
    pred_focals = pred_dict['focals'] # [b,n,1]
        
    with torch.no_grad():
        gt_hl = gt_hl.unsqueeze(1)
        gt_l, gt_r = compute_hl(gt_hl, dim=-1) # [b,1,3]

        pred_up_vectors = pred_frames[:,:,3:6]

        pred_hls = pred_up_vectors.clone()
        pred_hls[:,:,2:3] = pred_focals*pred_hls[:,:,2:3]
        
        est_ls, est_rs = compute_hl(pred_hls, dim=-1) # [b,n,3]

        err_ls = torch.abs(est_ls - gt_l)
        err_rs = torch.abs(est_rs - gt_r)

        hl_errors = torch.max(err_ls, err_rs) # 0 ~ inf
        hl_scores = hl_errors.neg().exp()

        up_scores = cosine_distance(pred_up_vectors, gt_up_vector, dim=2, keepdim=True)
                
        scores = (hl_scores + up_scores)/2.0
        scores = (-(((scores - 1.0)/sigma).pow(2)/2.0)).exp()    

        return scores

def compute_cands_cls(pred_dict, targets, alpha=0.5, sigma=0.5):        
    scores = compute_cands_score(pred_dict, targets, alpha=alpha, sigma=sigma)        
    gt_cls = (scores > 0.5).float()
    return gt_cls

class GPNetLoss():
    def __init__(self):
        super().__init__()
        self.bce_criterion = nn.BCEWithLogitsLoss(reduction='mean')
        self.cos_criterion = nn.CosineSimilarity(dim=1)
        self.sigmoid = nn.Sigmoid()

        self.thresh_p2p_pos = 1.0 - np.cos(np.radians(2.0), dtype=np.float32) # near 0.0
        self.thresh_p2p_neg = 1.0 - np.cos(np.radians(5.0), dtype=np.float32) # near 0.0

        self.thresh_l2p_pos = np.cos(np.radians(90.0 - 1.0), dtype=np.float32) # near 0.0
        self.thresh_l2p_neg = np.cos(np.radians(90.0 - 2.0), dtype=np.float32) # near 0.0

        self.total_loss = None

    def compute_angle_error(self, pred_dict, targets, stack_error=False):
        gt_up_vector = targets['gt_up_vector'].cuda()
        # gt_focal = targets['gt_focal'].cpu().numpy()
        gt_fov = targets['gt_vfov'].cpu().numpy()
        gt_rp = targets['gt_rp'].cpu().numpy()
        
        pred_up_vector = pred_dict['up_vector']
        pred_focal = pred_dict['focal'].cpu().numpy()
        pred_fov = 2.0*np.arctan2(1.0, pred_focal)

        num_samples = pred_up_vector.shape[0]

        cos_criterion = nn.CosineSimilarity(dim=0)
        
        if stack_error:
            total_rot_error =  []
            total_roll_error = []
            total_pitch_error = []
            total_fov_error = []
        else:
            total_rot_error = 0.0
            total_roll_error = 0.0
            total_pitch_error = 0.0
            total_fov_error = 0.0

        for i in range(num_samples):
            # gt_roll, gt_pitch = gt_rp[i]
            gt_pitch, gt_roll = decompose_up_vector(gt_up_vector[i])
            gt_pitch = gt_pitch.cpu().numpy()
            gt_roll = gt_roll.cpu().numpy()

            up_diff_cos = cos_criterion(pred_up_vector[i], gt_up_vector[i])

            if up_diff_cos.item() < 0.0:
                pred_pitch, pred_roll = decompose_up_vector(-pred_up_vector[i])        
            else:
                pred_pitch, pred_roll = decompose_up_vector(pred_up_vector[i])
            pred_pitch = pred_pitch.cpu().numpy()
            pred_roll = pred_roll.cpu().numpy()

            up_diff_cos = torch.abs(up_diff_cos)

            if stack_error:
                total_rot_error += [np.arccos(np.clip(up_diff_cos.item(), -1.0, 1.0))*180.0/math.pi]                
                total_pitch_error += [abs(pred_pitch - gt_pitch)*180.0/math.pi]
                total_roll_error += [abs(pred_roll - gt_roll)*180.0/math.pi]
                total_fov_error += [abs(pred_fov[i] - gt_fov[i])*180.0/math.pi]
            else:
                total_rot_error += np.arccos(np.clip(up_diff_cos.item(), -1.0, 1.0))*180.0/math.pi
                total_pitch_error += abs(pred_pitch - gt_pitch)*180.0/math.pi
                total_roll_error += abs(pred_roll - gt_roll)*180.0/math.pi
                total_fov_error += abs(pred_fov[i] - gt_fov[i])*180.0/math.pi
                
        return total_rot_error, total_roll_error, total_pitch_error, total_fov_error

    def compute_horizon_error_hlw(self, pred_dict, targets, stack_error=False):
        org_sz = targets['org_sz'].cpu().numpy()
        crop_sz = targets['crop_sz'].cpu().numpy()
        gt_hls = targets['gt_hl'].cpu().numpy()        
        
        est_up_vectors = pred_dict['up_vector']
        est_focals = pred_dict['focal']

        a,b,c = torch.split(est_up_vectors, 1, dim=1)
        est_hls = torch.cat([a,b,c*est_focals], dim=1)
        est_hls = est_hls.cpu().numpy()
        
        num_samples = est_hls.shape[0]

        if stack_error:
            total_horizon_error = []            
        else:
            total_horizon_error = 0.0

        for i in range(num_samples):
            gt_hl = hl_utils.compute_hl_hlw(gt_hls[i], org_sz[i])
            est_hl = hl_utils.compute_hl(est_hls[i], crop_sz[i], org_sz[i])

            err = compute_hl_error_np(gt_hl, est_hl, org_sz[i])

            if stack_error:
                total_horizon_error += [err]
            else:
                total_horizon_error += err

        return total_horizon_error

    def compute_horizon_error(self, pred_dict, targets, stack_error=False):
        org_sz = targets['org_sz'].cpu().numpy()
        gt_hls = targets['gt_hl'].cpu().numpy()        
        
        est_up_vectors = pred_dict['up_vector']
        est_focals = pred_dict['focal']

        a,b,c = torch.split(est_up_vectors, 1, dim=1)
        est_hls = torch.cat([a,b,c*est_focals], dim=1)
        est_hls = est_hls.cpu().numpy()
        
        num_samples = est_hls.shape[0]

        if stack_error:
            total_horizon_error = []            
        else:
            total_horizon_error = 0.0

        for i in range(num_samples):
            gt_hl = compute_hl_np(gt_hls[i], org_sz[i])
            est_hl = compute_hl_np(est_hls[i], org_sz[i])

            err = compute_hl_error_np(gt_hl, est_hl, org_sz[i])

            if stack_error:
                total_horizon_error += [err]
            else:
                total_horizon_error += err

        return total_horizon_error

    def compute_horizon(self, pred_dict, targets):
        org_sz = targets['org_sz'].cpu().numpy()
        gt_hls = targets['gt_hl'].cpu().numpy()        
        
        est_up_vectors = pred_dict['up_vector']
        est_focals = pred_dict['focal']

        a,b,c = torch.split(est_up_vectors, 1, dim=1)
        est_hls = torch.cat([a,b,c*est_focals], dim=1)
        est_hls = est_hls.cpu().numpy()

        num_samples = est_hls.shape[0]

        list_gt_hl = []
        list_est_hl = []

        for i in range(num_samples):
            gt_hl = compute_hl_np(gt_hls[i], org_sz[i])
            est_hl = compute_hl_np(est_hls[i], org_sz[i])

            list_gt_hl.append(gt_hl)
            list_est_hl.append(est_hl)

        return list_gt_hl, list_est_hl

    def compute_horizon_hlw(self, pred_dict, targets):
        org_sz = targets['org_sz'].cpu().numpy()
        crop_sz = targets['crop_sz'].cpu().numpy()
        gt_hls = targets['gt_hl'].cpu().numpy()        
        
        est_up_vectors = pred_dict['up_vector']
        est_focals = pred_dict['focal']

        a,b,c = torch.split(est_up_vectors, 1, dim=1)
        est_hls = torch.cat([a,b,c*est_focals], dim=1)
        est_hls = est_hls.cpu().numpy()

        num_samples = est_hls.shape[0]

        list_gt_hl = []
        list_est_hl = []

        for i in range(num_samples):
            gt_hl = hl_utils.compute_hl_hlw(gt_hls[i], org_sz[i])
            est_hl = hl_utils.compute_hl(est_hls[i], crop_sz[i], org_sz[i])

            list_gt_hl.append(gt_hl)
            list_est_hl.append(est_hl)

        return list_gt_hl, list_est_hl

    def compute_cands_score(self, pred_dict, targets, alpha=0.6, sigma=0.3):  
        return compute_cands_score(pred_dict, targets, alpha, sigma)
        
    def compute_cands_cls(self, pred_dict, targets, alpha=0.6, sigma=0.3):
        return compute_cands_cls(pred_dict, targets, alpha, sigma)
    
    def loss_st(self, logits, labels):
        with torch.no_grad():
            labels = _outer3(labels)
        
        logits= _outer3(logits)
        
        diff = (logits - labels).abs()        
        loss = diff.sum(dim=1)        
        return loss.mean()

    def loss_hl(self, pred_hl, gt_hl):
        with torch.no_grad():
            gt_l, gt_r = compute_hl(gt_hl)
        
        est_l, est_r = compute_hl(pred_hl)

        error_l = torch.abs(est_l - gt_l)
        error_r = torch.abs(est_r - gt_r)

        loss = torch.max(error_l, error_r)                
        return loss.mean()

    def vpts_cls_loss(self, gt_zvp, pred_vpts, pred_prob, out_dict):
        device = gt_zvp.device

        thresh_pos = 1.0 - np.cos(np.radians(2.0), dtype=np.float32) # near 0.0
        thresh_neg = 1.0 - np.cos(np.radians(5.0), dtype=np.float32) # near 0.0

        with torch.no_grad():
            gt_zvp = gt_zvp.unsqueeze(dim=1) # [b, 1, 3]            
            dist = 1.0 - cosine_distance(gt_zvp, pred_vpts) # [b, n] # near 0.0    
            
            const_0_0 = torch.tensor(0.0, device=device)
            const_1_0 = torch.tensor(1.0, device=device)    

            pos = torch.where(dist < thresh_pos, const_1_0, const_0_0)
            # neg = torch.where(dist > thresh_neg, const_1_0, const_0_0)
            # print('points pos', pos.sum().item(), 'neg', neg.sum().item())
            
            gt_prob = pos
            # gt_prob = torch.where(dist < thresh_pos, const_1_0, const_0_0)
            mask = torch.where(torch.gt(dist, thresh_pos) & torch.lt(dist, thresh_neg), 
                               const_0_0, const_1_0) # [b,n]
            

        print(pred_prob.size(), gt_prob.size())
        out_dict['n_pos'] = gt_prob.sum().item()
        out_dict['n_neg'] = (1.0 - gt_prob).sum().item()
        out_dict['n_ignore'] = (1.0 - mask).sum().item()
        
        pred_cls = torch.round(F.sigmoid(pred_prob))
        print(pred_cls.size())
        out_dict['accuracy'] = (pred_cls == gt_prob).sum().float()
        
        bce = nn.BCEWithLogitsLoss(reduction='none')
        pred_prob = pred_prob.squeeze(-1)
        losses = bce(pred_prob, gt_prob)
        loss = (mask*losses).mean()
        return loss

    def __call__(self, pred_dict, targets):
        out_dict = {}

        gt_up_vector = targets['gt_up_vector'].cuda()
        gt_zvp = targets['gt_zvp'].cuda()
        gt_hl = targets['gt_hl'].cuda()
        gt_focal = targets['gt_focal'].cuda()

        gt_cls = compute_cands_cls(pred_dict, targets, alpha=0.0, sigma=0.1)

        pred_vpts = pred_dict['vert/pts']
        pred_vpts_prob = pred_dict['vert/pts_prob']
        # pred_vlines = pred_dict['vert/lines']
        # pred_vlines_prob = pred_dict['vert/lines_prob']
        pred_zvp = pred_dict['zvp']
        
        pred_cls = pred_dict['logits']
                        
        # pred_up_vector = pred_dict['up_vector']
        # pred_focal = pred_dict['focal']

        # a,b,c = torch.split(pred_up_vector, 1, dim=1)
        # pred_hl = torch.cat([a,b,c*pred_focal], dim=1)

        total_loss = 0.0

        gt_zvp = F.normalize(gt_zvp, dim=1)
        pred_zvp = F.normalize(pred_zvp, dim=1)

        #gt_up_vector = F.normalize(gt_up_vector, dim=1)
        #pred_up_vector = F.normalize(pred_up_vector, dim=1)

        # loss = (1.0 - cosine_distance(gt_zvp, pred_zvp)).mean()
        loss = structure_tensor_loss(gt_zvp, pred_zvp, 1.0)
        # total_loss += loss
        out_dict['loss_zvp'] = loss.item()       
        
        loss = self.vpts_cls_loss(gt_zvp, pred_vpts, pred_vpts_prob, out_dict)
        # total_loss += loss
        out_dict['loss_vpts_cls'] = loss.item()
                
        loss = self.bce_criterion(pred_cls, gt_cls)
        total_loss += loss
        out_dict['loss_frame_score'] = loss.item()

        out_dict['n_gt_pos'] = gt_cls.sum()
        
        # # loss = (1.0 - torch.abs(cos_criterion(pred_up_vector, gt_up_vector))).mean()        
        # loss = structure_tensor_loss(pred_up_vector, gt_up_vector, 1.0)
        # total_loss += loss
        # out_dict['loss_up_vector'] = loss.item()
        
        # # loss = F.mse_loss(pred_focal, gt_focal)
        # # total_loss += loss
        # # out_dict['loss_focal'] = loss        
        
        # loss = self.loss_hl(pred_hl, gt_hl)
        # total_loss += loss
        # out_dict['loss_hl'] = loss.item()
                
        self.total_loss = total_loss
        out_dict['total_loss'] = total_loss.item()

        return total_loss.item(), out_dict

    def get_loss_var(self):
        return self.total_loss

if __name__ == "__main__":
    loss = GPNetLoss()
    print(loss)