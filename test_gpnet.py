from __future__ import division
import sys
import time
import numpy as np
import os

import torch
from torch.autograd import Variable
import models.networks_gpnet
from options.test_options_gpnet import TestOptions
from data import dataset_loader_gpnet
from models.models import create_model_gpnet
import random
import scipy.io
import csv

EVAL_BATCH_SIZE = 1
opt = TestOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch

root = '/'

def write_csv(filename, errors):
    with open(filename, 'w') as file:
        for err in errors:
            file.write(str(err)+'\n')

def test_numerical_angle_horizon(model, dataset, global_step, opt, name):
    rot_e_list = []
    roll_e_list = []
    pitch_e_list = []
    fov_e_list = []
    horizon_e_list = []
    
    model.switch_to_eval()

    for i, data in enumerate(dataset):
        stacked_img = data[0]
        stacked_line = data[1]
        stacked_seg_map = data[2]
        stacked_seg_mask = data[3]
        targets = data[4]

        (rotation_error, roll_error, pitch_error, fov_error, horizon_error) = (
            model.test_angle_horizon_error(stacked_img, stacked_line, 
                                           stacked_seg_map, stacked_seg_mask, targets))

        rot_e_list = rot_e_list + rotation_error	
        roll_e_list = roll_e_list + roll_error
        pitch_e_list = pitch_e_list + pitch_error
        fov_e_list = fov_e_list + fov_error
        horizon_e_list = horizon_e_list + horizon_error
        
        if i%100 == 0:
            print(i)

    rot_e_arr = np.array(rot_e_list)
    roll_e_arr = np.array(roll_e_list)
    pitch_e_arr = np.array(pitch_e_list)
    fov_e_arr = np.array(fov_e_list)
    horizon_e_arr = np.array(horizon_e_list)        

    mean_rot_e = np.mean(rot_e_arr)
    median_rot_e = np.median(rot_e_arr)
    std_rot_e = np.std(rot_e_arr)

    mean_roll_e = np.mean(roll_e_arr)
    median_roll_e = np.median(roll_e_arr)
    std_roll_e = np.std(roll_e_arr)

    mean_pitch_e = np.mean(pitch_e_arr)
    median_pitch_e = np.median(pitch_e_arr)
    std_pitch_e = np.std(pitch_e_arr)

    mean_fov_e = np.mean(fov_e_arr)
    median_fov_e = np.median(fov_e_arr)
    std_fov_e = np.std(fov_e_arr)

    mean_horizon_e = np.mean(horizon_e_arr)
    median_horizon_e = np.median(horizon_e_arr)
    std_horizon_e = np.std(horizon_e_arr)

    print('======================= FINAL STATISCIS ==========================')
    print('mean_rot_e {:0.02f}'.format(mean_rot_e))
    print('median_rot_e {:0.02f}'.format(median_rot_e))
    print('std_rot_e {:0.02f}'.format(std_rot_e))

    print('mean_pitch_e {:0.02f}'.format(mean_pitch_e))
    print('median_pitch_e {:0.02f}'.format(median_pitch_e))
    print('std_pitch_e {:0.02f}'.format(std_pitch_e))

    print('mean_roll_e {:0.02f}'.format(mean_roll_e))
    print('median_roll_e {:0.02f}'.format(median_roll_e))
    print('std_roll_e {:0.02f}'.format(std_roll_e))

    print('mean_fov_e {:0.02f}'.format(mean_fov_e))
    print('median_fov_e {:0.02f}'.format(median_fov_e))
    print('std_fov_e {:0.02f}'.format(std_fov_e))

    print('mean_horizon_e {:0.04f}'.format(mean_horizon_e))
    print('median_horizon_e {:0.04f}'.format(median_horizon_e))
    print('std_horizon_e {:0.04f}'.format(std_horizon_e))

    write_csv('gpnet_{}_{}.csv'.format(opt.dataset, name), horizon_e_list)

if __name__ == "__main__":

    eval_list_path = 'manhattan_filename_test_20200305.csv'
    
    eval_num_threads = 3

    test_data_loader = dataset_loader_gpnet.GSVDataLoader_gpnet(opt, eval_list_path, False, EVAL_BATCH_SIZE, eval_num_threads)
    test_dataset = test_data_loader.load_data()
    test_data_size = len(test_data_loader)
    print('========================= GoogleStreetView eval #images = %d ========='%test_data_size)

    model = create_model_gpnet(opt, _isTrain=False)
    model.switch_to_train()

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    global_step = 0

    test_numerical_angle_horizon(model, test_dataset, global_step, opt, opt.dataset)