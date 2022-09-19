from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from utils import *


def get_reference_data(config, data_type):
    raw_data = []
    exec(f"raw_data = read_data(config.{data_type}_data_path")
    assert raw_data != []
    raw_data = np.array(raw_data)
    data = torch.tensor(np.array([raw_data for _ in range(config.batch_size)]))
    data = data.reshape(config.batch_size, -1, 500, 500)
    data = data.to(config.device)
    return data


def get_query_dataloader(config):
    traffic_data_list = [os.path.join(config.DATA_PATH, f'traffic_speed/traffic_speed_{i}.pkl') for i in
                         range(0, 23374)]
    test_data_list_1 = [os.path.join(config.DATA_PATH, f'traffic_speed/traffic_speed_{i}.pkl') for i in
                        range(23374, 23772)]
    test_data_list_2, rest_data_list = train_test_split(traffic_data_list, train_size=0.18325, random_state=100)
    valid_data_list, train_data_list, = train_test_split(rest_data_list, train_size=0.25, random_state=100)
    test_data_list = test_data_list_1 + test_data_list_2
    if config.task == 'train':
        train_dataset = MaskDataset(read_data, train_data_list)
        valid_dataset = MaskDataset(read_data, valid_data_list)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0, drop_last=True)
        valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0, drop_last=True)
        return train_loader, valid_loader
    else:
        test_dataset = MaskDataset(read_data, test_data_list)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0, drop_last=True)
        return test_loader


class SpeedDataset(Dataset):
    def __init__(self, loader, file_list):
        self.inputs = file_list
        self.loader = loader

    def __getitem__(self, index):
        file = self.inputs[index]
        x = self.loader(file)['point']
        y = self.loader(file)['congestion']
        z = self.loader(file)['grid']
        return x, y, z

    def __len__(self):
        return len(self.inputs)


class SpeedClassDataset(Dataset):
    def __init__(self, loader, file_list):
        self.inputs = file_list
        self.loader = loader

    def __getitem__(self, index):
        file = self.inputs[index]
        x = self.loader(file)['point']
        y = self.loader(file)['speed']
        return x, y

    def __len__(self):
        return len(self.inputs)


class MaskDataset(Dataset):
    def __init__(self, loader, file_list):
        self.inputs = file_list
        self.loader = loader

    def __getitem__(self, index):
        file = self.inputs[index]
        point = self.loader(file)['point']
        x = generate_mask(point)
        y = self.loader(file)['congestion']
        z = self.loader(file)['grid']
        return x, y, z, point

    def __len__(self):
        return len(self.inputs)


class OnehotDataset(Dataset):
    def __init__(self, loader, file_list):
        self.inputs = file_list
        self.loader = loader

    def __getitem__(self, index):
        file = self.inputs[index]
        point = self.loader(file)['point']
        x = generate_onehot_mask(point)
        y = self.loader(file)['congestion']
        z = self.loader(file)['grid']
        return x, y, z, point

    def __len__(self):
        return len(self.inputs)


class BertDataset(Dataset):
    def __init__(self, loader, file_list):
        self.inputs = file_list
        self.loader = loader

    def __getitem__(self, index):
        file = self.inputs[index]
        x = self.loader(file)['data']
        y = self.loader(file)['data']
        return x, y

    def __len__(self):
        return len(self.inputs)
