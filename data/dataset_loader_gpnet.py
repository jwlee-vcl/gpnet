import sys
import random
import numpy as np
import torch.utils.data
from data.base_data_loader import BaseDataLoader
from data.image_folder_gpnet import *
from data import image_folder_hlw


class GSVDataLoader_gpnet(BaseDataLoader):
    def __init__(self, opt, list_path, is_train, _batch_size, num_threads):
        dat_basepath = opt.datapath
            
        dataset = GSVFolder_gpnet(opt=opt, 
            list_path=list_path, base_path=dat_basepath, is_train=is_train)
        self.data_loader = torch.utils.data.DataLoader(dataset, 
            batch_size=_batch_size, shuffle=is_train, num_workers=int(num_threads))
        self.dataset = dataset

    def load_data(self):
        return self.data_loader

    def name(self):
        return 'GSVDataLoader_gpnet'
    
    def __len__(self):
        return len(self.dataset)