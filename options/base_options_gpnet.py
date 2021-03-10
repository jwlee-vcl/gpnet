import argparse
import os
from util import util

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2')
        self.parser.add_argument('--name', type=str, default='test_local', help='name of the experiment. It decides where to store samples and models')

        self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints/', help='models are saved here')
 
        self.parser.add_argument('--log_comment', type=str, default='exp_gpnet_ls', help='tensorboard log dir comment')
        self.parser.add_argument('--dataset', type=str, required=True, help='Dataset for training')
        self.parser.add_argument('--mode', type=str, default='ResNet', help='Network models')

        # gpnet
        self.parser.add_argument('--thresh_line_len', type=float, default=10.0, help='minimum length of line segs')
        self.parser.add_argument('--n_in_lines', type=int, default=512, help='# of initial line segs')

        self.parser.add_argument('--seg_map_size', type=int, default=80, help='size of segments map')

        self.parser.add_argument('--n_vert_lines', type=int, default=256, help='# of initial vertical line segs')
        self.parser.add_argument('--n_vert_pts', type=int, default=256, help='# of initial vertical points')        

        self.parser.add_argument('--n_inlier_zvps', type=int, default=128, help='# of zenith vp candidates')        

        self.parser.add_argument('--n_in_vert_lines', type=int, default=256, help='# of initial vertical line segs')
        self.parser.add_argument('--n_in_hori_lines', type=int, default=512, help='# of initial horizion line segs')

        self.parser.add_argument('--n_juncs', type=int, default=512, help='# of junction points')

        self.parser.add_argument('--n_hori_lines', type=int, default=256, help='# of initial horizon line segs')
        self.parser.add_argument('--n_hori_pts', type=int, default=256, help='# of initial horizon points')        
        

        self.parser.add_argument('--ich_point', type=int, default=3, help='# of channel of input points')
        self.parser.add_argument('--ich_line', type=int, default=3, help='# of channel of input lines')
        self.parser.add_argument('--ich_frame', type=int, default=9, help='# of channel of input frames')

        self.parser.add_argument('--ch_inst', type=int, default=64, help='# of channel of instance feature')
        self.parser.add_argument('--ch_global', type=int, default=2048, help='# of channel of global feature')

        self.parser.add_argument('--datapath', type=str, help='path to dataset')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir =  os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
