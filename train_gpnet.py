from __future__ import division
import sys
import time
import numpy as np

import torch

import models.networks_gpnet
from options.train_options_gpnet import TrainOptions
from data.data_loader import *
from models.models import create_model_gpnet
import random
from tensorboardX import SummaryWriter

TRAIN_BATCH_SIZE = 2
EVAL_BATCH_SIZE = 2
opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch

print(opt.mode)

root = '/'

if __name__ == '__main__':

    summary_name = opt.dataset + '_runs/' + '%s_exp'%opt.log_comment + '_net_' + opt.mode + '_lr_' + str(opt.lr) 
    writer = SummaryWriter(summary_name)

    train_list_path = 'gsv_train_20200305.csv'
    eval_list_path = 'gsv_val_20200305.csv'

    train_num_threads = 3
    train_data_loader = GSVDataLoader_gpnet(opt, train_list_path, True, TRAIN_BATCH_SIZE, train_num_threads)
    train_dataset = train_data_loader.load_data()
    train_data_size = len(train_data_loader)
    print('========================= GoogleStreetView training #images = %d ========='%train_data_size)

    iteration_per_epoch = train_data_size//TRAIN_BATCH_SIZE
    eval_num_threads = 3

    test_data_loader = GSVDataLoader_gpnet(opt, eval_list_path, False, EVAL_BATCH_SIZE, eval_num_threads)
    test_dataset = test_data_loader.load_data()
    test_data_size = len(test_data_loader)
    print('========================= GoogleStreetView eval #images = %d ========='%test_data_size)

    iteration_per_epoch_eval = test_data_size//EVAL_BATCH_SIZE

    model = create_model_gpnet(opt, True)
    model.switch_to_train()
    model.set_writer(writer)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    global_step = 0


    def validation_numerical(model, dataset, global_step):
        total_horizon_error = 0.0
        
        model.switch_to_eval()

        count = 0

        upper_bound = 200

        rad_summary_idx = np.random.randint(0, min(upper_bound, iteration_per_epoch_eval-1))

        for i, data in enumerate(dataset):
            stacked_img = data[0]
            stacked_line = data[1]
            stacked_seg_map = data[2]
            stacked_seg_mask = data[3]
            targets = data[4]

            if i == rad_summary_idx:
                write_summary = True
            else:
                write_summary = False

            horizon_error = model.evaluate_horizon(stacked_img, stacked_line, stacked_seg_map, stacked_seg_mask, targets, global_step, write_summary)

            total_horizon_error += horizon_error
                        
            count += stacked_img.size(0)

            if i > upper_bound:
                break

        avg_horizon_error = float(total_horizon_error)/float(count)
        
        print('iteration_per_epoch_eval ', iteration_per_epoch_eval)
        
        print('============== avg_horizon_error: %d %f'%(i, avg_horizon_error))
        
        model.writer.add_scalar('Eval/avg_horizon_error', avg_horizon_error, global_step)
        
        model.switch_to_train()

        return avg_horizon_error

    eval_interval = iteration_per_epoch//6
    
    stop_epoch = 30

    for epoch in range(stop_epoch):

        model.update_learning_rate()

        for i, data in enumerate(train_dataset):
            global_step = global_step + 1
            
            if i%100 == 0:
                print('global_step', global_step)
            stacked_img = data[0]
            stacked_line = data[1]
            stacked_seg_map = data[2]
            stacked_seg_mask = data[3]
            targets = data[4]

            model.set_input(stacked_img, stacked_line, stacked_seg_map, stacked_seg_mask, targets)
            model.optimize_parameters(global_step)

            if global_step%eval_interval == 0:
                model.save('checkpoint_%s_%s_mode_'%(opt.dataset, opt.log_comment) + opt.mode + '_lr_' + str(opt.lr))

    print('we are done, stop training !!!')
        