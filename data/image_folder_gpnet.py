import os
import os.path
import sys, traceback
import torch
import torch.utils.data as data
import pickle
import glob

import time

import numpy as np
import numpy.linalg as LA

from scipy import misc 
import scipy.io as sio

import PIL
from PIL import Image, ImageEnhance
import math, random

import cv2 as cv
import skimage
import skimage.draw as skdraw

import pandas as pd

from skimage.transform import rotate
from skimage.morphology import remove_small_objects

import matplotlib.pyplot as plt

# thr = 0.01*np.power(1.75,np.arange(20))
# lsd = cv.createLineSegmentDetector(cv.LSD_REFINE_ADV, _log_eps=thr[5])
lsd = cv.createLineSegmentDetector(0)

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

def center_crop(img):
    sz = img.shape[0:2]
    side_length = np.min(sz)
    if sz[0] > sz[1]:
        ul_x = 0  
        ul_y = int(np.floor((sz[0]/2) - (side_length/2)))
        x_inds = [ul_x, sz[1]-1]
        y_inds = [ul_y, ul_y + side_length - 1]
    else:
        ul_x = int(np.floor((sz[1]/2) - (side_length/2)))
        ul_y = 0
        x_inds = [ul_x, ul_x + side_length - 1]
        y_inds = [ul_y, sz[0]-1]

    c_img = img[y_inds[0]:y_inds[1]+1, x_inds[0]:x_inds[1]+1]

    return c_img

def read_line_file(filename, thresh_len_px=10):
    segs = pd.read_csv(filename, sep=',', header=None).to_numpy()    
    segs = np.float32(segs)
    lens = LA.norm(segs[:,2:] - segs[:,:2], axis=1)
    segs = segs[lens > thresh_len_px]
    return segs

def normalize_segs(segs, pp, rho):    
    pp = np.array([pp[0], pp[1], pp[0], pp[1]], dtype=np.float32)    
    return rho*(segs - pp)

def sample_segs(segs, size, do_square=False):
    # linear prob? squared prob?
    lens = LA.norm(segs[:,2:] - segs[:,:2], axis=-1)
    if do_square:
        lens = np.square(lens)
    prob = lens/np.sum(lens)
        
    idx = np.random.choice(segs.shape[0], size, replace=True, p=prob)    
    sampled_segs = segs[idx]

    return sampled_segs

def cvPoint(pt):
    return (int(pt[0]), int(pt[1]))

def draw_seg(img, pt1, pt2, color):
    h,w = img.shape[:2]
    (ret, pt1, pt2) = cv.clipLine((0,0,w,h), cvPoint(pt1), cvPoint(pt2))
    if ret == False:
        return
    rr, cc = skdraw.line(int(pt1[1]), int(pt1[0]), int(pt2[1]), int(pt2[0]))    
    img[rr, cc] = color

def safe_normalize_np(v, eps=1e-6):
    de = LA.norm(v, axis=-1, keepdims=True)
    de = np.maximum(de, eps)
    return v/de

def segs2lines_np(segs, focal=1.0):
    focal = np.repeat(focal, len(segs))
    focal = np.expand_dims(focal, axis=-1)
    p1 = np.concatenate([segs[:,:2], focal], axis=-1)
    p2 = np.concatenate([segs[:,2:], focal], axis=-1)
    lines = np.cross(p1, p2)
    return safe_normalize_np(lines)

def cos_dist_np(v1, v2):
    v1 = safe_normalize_np(v1)
    v2 = safe_normalize_np(v2)
    return np.abs(np.sum(v1*v2, axis=-1))

def filter_segs_np(up_vector, focal, segs, thresh=3.0):
    '''
    up_vector [3] 
    focal 1
    segs [n,4]
    '''
    thresh = np.cos(np.radians(90.0 - thresh))

    lines = segs2lines_np(segs, focal) # [n,3]    
    dists = cos_dist_np(up_vector, lines)
    
    vert_segs = []
    hori_segs = []
    for i in range(len(dists)):
        if dists[i] < thresh:
            vert_segs.append(segs[i])
        else:
            hori_segs.append(segs[i])
    return np.array(vert_segs), np.array(hori_segs)

def generate_segs_map_np(segs, size=224):
    img = np.ones([size,size,4], dtype=np.float32)
    mask = np.zeros([size,size,1], dtype=np.float32)
    if not len(segs) > 0:
        return img, mask
        
    lens = LA.norm(segs[:,2:] - segs[:,:2], axis=-1)
    ind = np.argsort(lens)

    scale = size/2
    colors = segs.copy()
    segs = segs*scale    

    center = [size/2, size/2]
    center = np.array([center[0],center[1],center[0],center[1]], dtype=np.float32)
    segs = segs + center
    
    segs = segs[ind].copy()
    colors = colors[ind].copy()
    
    for i in range(len(segs)):
        draw_seg(img, segs[i,:2], segs[i,2:], color=colors[i])
        draw_seg(mask, segs[i,:2], segs[i,2:], color=1.0)
    return img, mask

class GSVFolder_gpnet(data.Dataset):
    def __init__(self, opt, list_path, base_path, is_train):
        self.base_path = base_path
        self.list_path = list_path		
        self.opt = opt
        self.input_size = 224        
        # self.seg_map_size = opt.seg_map_size

        self.is_train = is_train

        self.list_filename = []
        self.list_img_filename = []
        self.list_line_filename = []
        self.list_pitch = []
        self.list_roll = []
        self.list_focal = []

        values = pd.read_csv(self.list_path, sep=',', header=None).to_numpy()
        for row in values:
            img_filename  = self.base_path + row[0]
            line_filename  = self.base_path + row[1]
            self.list_filename.append(row[0])
            self.list_img_filename.append(img_filename)
            self.list_line_filename.append(line_filename)

        self.list_pitch = np.float32(values[:,3])
        self.list_roll  = np.float32(values[:,4])
        self.list_focal = np.float32(values[:,5])

    def __getitem__(self, index):
        targets_1 = {}

        filename = self.list_filename[index]
        # read image and preprocess
        img_path = self.list_img_filename[index]     
        line_path = self.list_line_filename[index]
        
        tic = time.time()
        img = cv.imread(img_path)
        # assert img is not None, print(img_path)
        org_img = img

        org_h, org_w = img.shape[0], img.shape[1]
        org_sz = np.array([org_h, org_w]) 
        crop_sz = np.array([org_h, org_w]) 
        
        img = cv.resize(img, dsize=(self.input_size, self.input_size))
        input_sz = np.array([self.input_size, self.input_size])

        img_1 = np.float32(img)/255.0        

        gray_img = cv.cvtColor(org_img, cv.COLOR_BGR2GRAY)        
        segs_org = lsd.detect(gray_img)[0]
        segs_org = segs_org[:,0,:]        

        pp = (org_w/2, org_h/2)
        rho = 2.0/np.maximum(org_w,org_h)

        segs = normalize_segs(segs_org, pp, rho)
        segs_sampled = sample_segs(segs, size=self.opt.n_in_lines)
        
        # generate seg map
        seg_map, seg_mask = generate_segs_map_np(segs, size=self.input_size)
        toc = time.time() - tic

        # preprocess GT data
        gt_pitch = np.radians(self.list_pitch[index])
        gt_roll = np.radians(self.list_roll[index])
        gt_focal = rho*self.list_focal[index]

        rotm = eul2rotm_ypr([gt_pitch, 0, gt_roll])
        rotm[1,:] = -rotm[1,:]
        gt_up_vector = rotm[:,1]
        
        gt_hl = gt_up_vector.copy()
        gt_hl[2] = gt_focal*gt_hl[2]

        gt_zvp = gt_up_vector.copy()
        if gt_zvp[2] < 0:
            gt_zvp = -gt_zvp
        gt_zvp = gt_zvp / np.maximum(gt_zvp[2], 1e-7)
        gt_zvp = gt_focal*gt_zvp
        gt_zvp[2] = 1.0

        ratio_x = float(self.input_size)/float(org_w)
        ratio_y = float(self.input_size)/float(org_h)

        fx_px = gt_focal * ratio_x
        fy_px = gt_focal * ratio_y

        gt_rp = np.array([gt_roll, gt_pitch]) 

        gt_vfov = 2.0 * np.arctan(1.0/gt_focal)

        offset = np.tan(-gt_pitch) / np.tan(gt_vfov/2.0) # h = 2
        slope = gt_roll
        
        final_img = torch.from_numpy(np.ascontiguousarray(img_1).transpose(2,0,1)).contiguous().float()
        final_segs = torch.from_numpy(np.ascontiguousarray(segs_sampled)).float()
        final_seg_map = torch.from_numpy(np.ascontiguousarray(seg_map)).contiguous().float()
        final_seg_mask = torch.from_numpy(np.ascontiguousarray(seg_mask)).contiguous().float()
        targets_1['gt_rp'] = torch.from_numpy(np.ascontiguousarray(gt_rp)).contiguous().float()
        targets_1['gt_vfov'] = torch.from_numpy(np.ascontiguousarray(gt_vfov)).contiguous().float()        
        targets_1['gt_up_vector'] = torch.from_numpy(np.ascontiguousarray(gt_up_vector)).contiguous().float()
        targets_1['gt_focal'] = torch.from_numpy(np.ascontiguousarray(gt_focal)).contiguous().float()
        targets_1['gt_zvp'] = torch.from_numpy(np.ascontiguousarray(gt_zvp)).contiguous().float()
        targets_1['gt_hl'] = torch.from_numpy(np.ascontiguousarray(gt_hl)).contiguous().float()
        targets_1['img_path'] = img_path
        targets_1['gt_slope'] = slope
        targets_1['gt_offset'] = offset
        targets_1['org_sz'] = org_sz
        targets_1['crop_sz'] = crop_sz
        targets_1['input_sz'] = input_sz
        targets_1['filename'] = filename
        targets_1['org_img'] = org_img
        targets_1['seg_map'] = seg_map
        targets_1['seg_mask'] = seg_mask

        return final_img, final_segs, final_seg_map, final_seg_mask, targets_1

    def __len__(self):
        return len(self.list_img_filename)
