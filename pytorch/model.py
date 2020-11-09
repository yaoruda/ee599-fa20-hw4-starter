from torchvision.models import resnet50, resnet34
import torch.nn.functional as F
import torch.nn as nn

from utils import local, Config


if local:
    torch_model = resnet50(pretrained=Config['finetune'])
else:
    torch_model = resnet34(pretrained=Config['finetune'])


class Ruda_Model(nn.Module):
    def __init__(self):
        super(Ruda_Model, self).__init__()

        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4_1 = nn.Conv2d(256, 384, 3, padding=1)
        self.conv4_2 = nn.Conv2d(384, 384, 3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(384)
        self.bn4_2 = nn.BatchNorm2d(384)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5_1 = nn.Conv2d(384, 256, 3, padding=1)
        self.conv5_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(256)
        self.bn5_2 = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(kernel_size=2, dilation=2)

        self.dropout_conv = nn.Dropout2d(0.1)
        self.dropout_fc = nn.Dropout2d(0.5)

        self.fc1 = nn.Linear(256 * 6 * 6, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 153)

    def forward(self, x):
        out = x

        out = self.conv1_1(out)
        out = self.bn1_1(out)
        out = self.conv1_2(out)
        out = self.bn1_2(out)
        out = self.pool1(out)
        out = F.relu(out)
        out = self.dropout_conv(out)

        out = self.conv2_1(out)
        out = self.bn2_1(out)
        out = self.conv2_2(out)
        out = self.bn2_2(out)
        out = self.pool2(out)
        out = F.relu(out)
        out = self.dropout_conv(out)

        out = self.conv3_1(out)
        out = self.bn3_1(out)
        out = self.conv3_2(out)
        out = self.bn3_2(out)
        out = self.pool3(out)
        out = F.relu(out)
        out = self.dropout_conv(out)

        out = self.conv4_1(out)
        out = self.bn4_1(out)
        out = self.conv4_2(out)
        out = self.bn4_2(out)
        out = self.pool4(out)
        out = F.relu(out)
        out = self.dropout_conv(out)

        out = self.conv5_1(out)
        out = self.bn5_1(out)
        out = self.conv5_2(out)
        out = self.bn5_2(out)
        out = self.pool5(out)
        out = F.relu(out)
        out = self.dropout_conv(out)

        # print(out.shape)
        out = out.view(-1, 256 * 6 * 6)

        out = self.fc1(out)
        out = self.dropout_fc(out)
        out = self.fc2(out)
        out = self.dropout_fc(out)
        out = self.fc3(out)

        return out
