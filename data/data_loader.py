from data.dataset_loader import *
from data.dataset_loader_gpnet import *

def CreateGSVDataLoader_gpnet(opt, list_path, is_train, _batch_size, num_threads):
    data_loader = GSVDataLoader_gpnet(opt, list_path, is_train, _batch_size, num_threads)
    return data_loader    