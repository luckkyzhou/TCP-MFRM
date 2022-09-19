from model import *
from dataset import *
from utils import *
from config import *


def test(config):
    house_data = get_reference_data(config, 'house')
    infras_data = get_reference_data(config, 'infras')
    micro_data = get_reference_data(config, 'micro')

    test_loader = get_query_dataloader(config)

    model = FusionNet(house_data.shape[1], infras_data.shape[1], micro_data.shape[1], config.encoder_channels, config.grid_channels)
    if torch.device == 'cpu':
        model.load_state_dict(torch.load(config.model_path, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(config.model_path))
    model = model.to(config.device)

    criterion = nn.MSELoss()

    test_loss = 0.0
    test_accuracy = 0.0
    FN = 0
    FP = 0
    TP = 0
    TN = 0

    for x, y, _, _ in test_loader:
        x, y = x.float(), y.float()
        x, y = x.to(config.device), y.to(config.device)

        with torch.no_grad():
            y_pred = model(house_data, infras_data, micro_data, x)
            loss = criterion(y_pred, y)

        test_loss += loss.item()
        test_accuracy += calculate_accuracy(y_pred, y)
        correct_01, correct_10, correct_11, correct_00 = count_metrics(y_pred, y)
        FN += correct_01
        FP += correct_10
        TP += correct_11
        TN += correct_00
    test_total_loss = test_loss / len(test_loader)
    test_total_accuracy = test_accuracy / len(test_loader)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    f1 = 2 * precision * recall / (precision + recall)
    print('Test Loss = {:.8f}, Test Precision = {:.8f}, Test Recall = {:.8f}, Test Accuracy = {:.8f}, Test F1 = {:.8f},'
          .format(test_total_loss, precision, recall, accuracy, f1))
    save_log(config, 'Test Loss = {:.8f}, Test Precision = {:.8f}, Test Recall = {:.8f}, Test Accuracy = {:.8f}, Test F1 = {:.8f},'
             .format(test_total_loss, precision, recall, accuracy, f1))


if __name__ == '__main__':
    args = get_args()
    config = Config(args)

    test(config)
