import torch
import torch.nn as nn


class FusionNet(nn.Module):
    def __init__(self, c1, c2, c3, encoder_channels, grid_channels):
        super(FusionNet, self).__init__()
        self.conv1 = ConvNet(c1, 16, 32)
        self.conv2 = ConvNet(c2, 16, 32)
        self.conv3 = ConvNet(c3, 16, 32)
        self.encoder = Encoder(32*3, encoder_channels)
        self.gridFC = GridFC(grid_channels)
        self.decoder = Decoder(encoder_channels + grid_channels)
        self.features = torch.tensor([])

    def forward(self, x1, x2, x3, x4):
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        x = torch.cat([x1, x2, x3], 1)
        z = self.encoder(x)
        self.features = z
        x4 = self.gridFC(x4)
        x = torch.cat([z, x4], 1)
        x = self.decoder(x)
        return x


class GridFC(nn.Module):
    def __init__(self, output_channels):
        super(GridFC, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.conv3 = nn.Conv2d(64, 128, 5, padding=2)
        self.convTrans1 = nn.ConvTranspose2d(128, 64, 5, stride=2, padding=2, output_padding=1)
        self.convTrans2 = nn.ConvTranspose2d(64, output_channels, 5, stride=2, padding=2, output_padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(output_channels)
        self.pool = nn.AvgPool2d(2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.bn3(x)
        x = self.convTrans1(x)
        x = self.bn2(x)
        x = self.convTrans2(x)
        x = self.bn4(x)
        return x


class ConvNet(nn.Module):
    def __init__(self, input_channels, middel_channels, output_channels):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, middel_channels, 5, padding=2)
        self.conv2 = nn.Conv2d(middel_channels, output_channels, 5, padding=2)
        self.bn1 = nn.BatchNorm2d(middel_channels)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.relu(self.conv2(x))
        x = self.bn2(x)
        return x


class Encoder(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.conv3 = nn.Conv2d(64, 128, 5, padding=2)
        self.convTrans1 = nn.ConvTranspose2d(128, 64, 5, stride=2, padding=2, output_padding=1)
        self.convTrans2 = nn.ConvTranspose2d(64, output_channels, 5, stride=2, padding=2, output_padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(output_channels)
        self.pool = nn.AvgPool2d(2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.bn3(x)
        x = self.convTrans1(x)
        x = self.bn2(x)
        x = self.convTrans2(x)
        x = self.bn4(x)
        return x


class Decoder(nn.Module):
    def __init__(self, input_channels):
        super(Decoder, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv2d(input_channels, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.fc1 = nn.Linear(64*125*125, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 216)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(1024)
        self.pool = nn.AvgPool2d(2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.bn3(x)
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.bn4(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x
