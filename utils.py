import os
import numpy as np
import itertools
import pickle
import scipy.stats as st
import torch
import math


def one_hot_encode(label, label_names):
    encoded = np.zeros(len(label_names))
    encoded[label_names.index(label)] = 1

    return np.array(encoded, dtype='float32')


def split_dict(d, slice_len):
    out_list = list()
    if len(d) % slice_len == 0:
        n = len(d) // slice_len
    else:
        n = len(d) // slice_len + 1
    i = iter(d.items())
    for _ in range(n):
        d = dict(itertools.islice(i, slice_len))
        out_list.append(d)
    return out_list


def save_log(config, log):
    file_path = config.log_path
    with open(file_path, 'a') as f:
        f.write(log)


def save_data(config, dataset, file_name, sub_path=''):
    save_file_name = os.path.join(config.DATA_PATH, sub_path, file_name + '.pkl')
    with open(save_file_name, 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
    print(f"{save_file_name} saved.")


def read_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def generate_gaussian_kernel(kernel_size, sigma):
    """Returns a 2D Gaussian kernel."""
    x = np.linspace(-sigma, sigma, kernel_size + 1)
    kernel_1d = np.diff(st.norm.cdf(x))
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    kernel_2d = kernel_2d / kernel_2d.sum()
    kernel_2d = kernel_2d / kernel_2d.max()
    return kernel_2d


G_KERNEL = generate_gaussian_kernel(999, 3)


def generate_mask(point):
    i, j = point
    i, j = int(i), int(j)
    mask = G_KERNEL[499-i:999-i, 499-j:999-j]
    mask = mask.reshape(1, 500, 500)
    mask = torch.tensor(mask, dtype=torch.float32)
    return mask


def generate_onehot_mask(point):
    i, j = point
    i, j = int(i), int(j)
    mask = np.zeros((500, 500, 1))
    mask[i][j][0] = 1
    mask = mask.reshape(1, 500, 500)
    mask = torch.tensor(mask, dtype=torch.float32)
    return mask


def calculate_accuracy(y_pred, y):
    accuracy = 0.0
    cnt = 0
    for i in range(y.shape[0]):
        corrects = 0
        for j in range(y.shape[1]):
            if (y_pred[i][j] >= 0.5 and y[i][j] == 1) or (y_pred[i][j] < 0.5 and y[i][j] == 0):
                corrects += 1
        accuracy += corrects / y.shape[1]
        cnt += 1
    return accuracy / cnt


def count_metrics(y_pred, y):
    correct_01 = 0
    correct_10 = 0
    correct_11 = 0
    correct_00 = 0

    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            if y_pred[i][j] >= 0.5 and y[i][j] == 1:
                correct_11 += 1
            elif y_pred[i][j] < 0.5 and y[i][j] == 0:
                correct_00 += 1
            elif y_pred[i][j] >= 0.5 and y[i][j] == 0:
                correct_10 += 1
            else:
                correct_01 += 1
    return correct_01, correct_10, correct_11, correct_00


def count_metrics_one(y_pred, y):
    correct_01 = 0
    correct_10 = 0
    correct_11 = 0
    correct_00 = 0

    for i in range(y.shape[0]):
        if y_pred[i] >= 0.5 and y[i] == 1:
            correct_11 += 1
        elif y_pred[i] < 0.5 and y[i] == 0:
            correct_00 += 1
        elif y_pred[i] >= 0.5 and y[i] == 0:
            correct_10 += 1
        else:
            correct_01 += 1
    return correct_01, correct_10, correct_11, correct_00


def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i])**2
    return math.sqrt(distance)


def get_neighbors(point, train_dataloader, num_neighbors):
    distances = list()
    for x, y, z in train_dataloader:
        x = x[0]
        y = y[0]
        dist = euclidean_distance(point, x)
        distances.append((x, y, dist))
    distances.sort(key=lambda tup: tup[2])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i])
    return neighbors


def predict_classification(point, train_dataloader, num_neighbors):
    neighbors = get_neighbors(point, train_dataloader, num_neighbors)
    predictions = list()
    for i in range(216):
        output_values = [neighbor[1][i] for neighbor in neighbors]
        prediction = max(set(output_values), key=output_values.count)
        predictions.append(int(prediction))
    return torch.tensor(predictions)
