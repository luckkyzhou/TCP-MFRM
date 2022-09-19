import os
import argparse
import torch
import time


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', required=True, type=str)
    parser.add_argument('--epoch_nums', type=int, default=20, help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size.')
    parser.add_argument('--encoder_channels', type=int, default=64)
    parser.add_argument('--grid_channels', type=int, default=16)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--task', required=True, type=str, default='train', help='Train or Test.')
    parser.add_argument('--model_name', type=str, default='')
    return parser.parse_args()


class Config:
    def __init__(self, config):
        self.root_path = config.root_path
        self.epoch_nums = config.epoch_nums
        self.batch_size = config.batch_size
        self.encoder_channels = config.encoder_channels
        self.grid_channels = config.grid_channels
        self.device = config.gpu if torch.cuda.is_available() else 'cpu'

        self.DATA_PATH = os.path.join(self.root_path, 'data/')
        self.MODEL_PATH = os.path.join(self.root_path, 'model/')
        self.LOG_PATH = os.path.join(self.root_path, 'log/')
        path_list = [self.DATA_PATH, self.MODEL_PATH, self.LOG_PATH]
        for path in path_list:
            if not os.path.exists(path):
                os.makedirs(path)
        self.house_data_path = os.path.join(self.DATA_PATH, 'house_price.pkl')
        self.infras_data_path = os.path.join(self.DATA_PATH, 'infrastructure.pkl')
        self.micro_data_path = os.path.join(self.DATA_PATH, 'micro_bert.pkl')
        if config.model_name:
            self.model_path = os.path.join(self.MODEL_PATH, f"{config.model_name}.pt")
            self.log_path = os.path.join(self.LOG_PATH, f"{config.model_name}_test_result.txt")
        else:
            self.model_path_best = os.path.join(self.MODEL_PATH, "TCP-MFRM_" + time.strftime("%Y%m%d_%H%M%S", time.localtime()) + "_best.pt")
            self.model_path_last = os.path.join(self.MODEL_PATH, "TCP-MFRM_" + time.strftime("%Y%m%d_%H%M%S",
                                                                                             time.localtime()) + "_last.pt")
            self.log_path = os.path.join(self.LOG_PATH, "TCP-MFRM_" + time.strftime("%Y%m%d_%H%M%S", time.localtime()) + ".txt")

        self.lng_min = 116.1186218
        self.lng_max = 116.6802978
        self.lat_min = 39.7145817
        self.lat_max = 40.1626081
