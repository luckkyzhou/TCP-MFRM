import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR

from config import *
from model import *
from dataset import *
from utils import *


def train(config):
    house_data = get_reference_data(config, 'house')
    infras_data = get_reference_data(config, 'infras')
    micro_data = get_reference_data(config, 'micro')

    train_loader, valid_loader = get_query_dataloader(config)

    model = FusionNet(house_data.shape[1], infras_data.shape[1], micro_data.shape[1], config.encoder_channels, config.grid_channels)
    model = model.to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = MultiStepLR(optimizer, milestones=[3, 12], gamma=0.1)
    criterion = nn.MSELoss()

    max_valid_accuracy = -np.inf
    print("Start Training......")
    for epoch in range(config.epoch_nums):
        print(f"Epoch {epoch} starts ......")
        train_iter = 0  # count for iters of current epoch
        len_iters = len(train_loader) - 1  # total iters of current epoch
        for x, y, z, _ in train_loader:
            x, y = x.float(), y.float()
            x, y = x.to(config.device), y.to(config.device)

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                y_pred = model(house_data, infras_data, micro_data, x)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
            train_iter_loss = loss.item()  # loss of current iter
            train_iter_accuracy = calculate_accuracy(y_pred, y)
            print('Epoch {} - Iteration {}/{}: Training Accuracy = {:.4f}, Training Loss = {:.8f},'.format(
                epoch, train_iter, len_iters, train_iter_accuracy, train_iter_loss))
            save_log(config, 'Epoch {} - Iteration {}/{}: Training Accuracy = {:.4f},  Training Loss = {:.8f},\n'.format(
                epoch, train_iter, len_iters, train_iter_accuracy, train_iter_loss))
            train_iter += 1

        valid_loss = 0.0
        valid_accuracy = 0.0
        for x, y, _, _ in valid_loader:
            x, y = x.float(), y.float()
            x, y = x.to(config.device), y.to(config.device)

            optimizer.zero_grad()
            with torch.no_grad():
                y_pred = model(house_data, infras_data, micro_data, x)
                loss = criterion(y_pred, y)
            valid_loss += loss.item()
            valid_accuracy += calculate_accuracy(y_pred, y)
        valid_epoch_loss = valid_loss / len(valid_loader)
        valid_epoch_accuracy = valid_accuracy / len(valid_loader)
        print('Epoch {} - Valid Accuracy = {:.4f}, Valid Loss = {:.8f},'.format(epoch, valid_epoch_accuracy,
                                                                                valid_epoch_loss))
        save_log(config, 'Epoch {} - Valid Accuracy = {:.4f},  Valid Loss = {:.8f},\n'.format(
            epoch, valid_epoch_accuracy, valid_epoch_loss))

        scheduler.step()

        if max_valid_accuracy < valid_epoch_accuracy:
            print(f"Validation accuracy increased: {max_valid_accuracy:.4f} --> {valid_epoch_accuracy:.4f}")
            save_log(config, f"Validation accuracy increased: {max_valid_accuracy:.4f} --> {valid_epoch_accuracy:.4f},\n")
            max_valid_accuracy = valid_epoch_accuracy
            torch.save(model.state_dict(), config.model_path_best)
        torch.save(model.state_dict(), config.model_path_last)


def main():
    args = get_args()
    config = Config(args)

    train(config)
