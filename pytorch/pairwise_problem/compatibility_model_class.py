import torch.nn.functional as F
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv1_1 = nn.Conv2d(6, 64, 3, bias=False)
        self.conv1_2 = nn.Conv2d(64, 64, 3, bias=False)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, dilation=2)

        self.conv2_1 = nn.Conv2d(64, 128, 3, bias=False)
        self.conv2_2 = nn.Conv2d(128, 128, 3, bias=False)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, dilation=2)

        self.conv3_1 = nn.Conv2d(128, 256, 3, bias=False)
        self.conv3_2 = nn.Conv2d(256, 256, 3, bias=False)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, dilation=2)

        self.conv4_1 = nn.Conv2d(256, 512, 3, bias=False)
        self.conv4_2 = nn.Conv2d(512, 512, 3, bias=False)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, dilation=2)

        self.conv5_1 = nn.Conv2d(512, 1024, 3, padding=1)
        self.conv5_2 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(1024)
        self.bn5_2 = nn.BatchNorm2d(1024)
        self.pool5 = nn.MaxPool2d(kernel_size=2, dilation=2)

        self.dropout = nn.Dropout2d(0.1)

        self.fc1 = nn.Linear(1024 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 1)


    def forward(self, x):

        out = x

        out = self.conv1_1(out)
        out = self.bn1_1(out)
        out = self.conv1_2(out)
        out = self.bn1_2(out)
        out = self.pool1(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.conv2_1(out)
        out = self.bn2_1(out)
        out = self.conv2_2(out)
        out = self.bn2_2(out)
        out = self.pool2(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.conv3_1(out)
        out = self.bn3_1(out)
        out = self.conv3_2(out)
        out = self.bn3_2(out)
        out = self.pool3(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.conv4_1(out)
        out = self.bn4_1(out)
        out = self.conv4_2(out)
        out = self.bn4_2(out)
        out = self.pool4(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.conv5_1(out)
        out = self.bn5_1(out)
        out = self.conv5_2(out)
        out = self.bn5_2(out)
        out = self.pool5(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = out.view(-1, 1024 * 4 * 4)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

# model = Model()
