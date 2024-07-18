import torch
import torch.nn as nn
import torchvision.models as models

# res_blockの実装
class Block(nn.Module):
    def __init__(self, first_conv_in_channels, first_conv_out_channels, identity_conv=None, stride=1):
        super(Block, self).__init__()

        self.conv1 = nn.Conv2d(
            first_conv_in_channels, first_conv_out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(first_conv_out_channels)

        self.conv2 = nn.Conv2d(
            first_conv_out_channels, first_conv_out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(first_conv_out_channels)

        self.conv3 = nn.Conv2d(
            first_conv_out_channels, first_conv_out_channels*4, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(first_conv_out_channels*4)
        self.relu = nn.ReLU()

        self.identity_conv = identity_conv
    
    def forward(self, x):

        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_conv is not None:
            identity = self.identity_conv(identity)
        x += identity

        x = self.relu(x)

        return x
    

#ResNet-50の実装
class ResNet(nn.Module):
    def __init__(self, block, num_classes):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(271, 300, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(300)
        self.relu = nn.ReLU()
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2_x = self._make_layer(block, 3, res_block_in_channels=300, first_conv_out_channels=300, stride=1)

        self.conv3_x = self._make_layer(block, 4, res_block_in_channels=1200,  first_conv_out_channels=600, stride=1)
        self.conv4_x = self._make_layer(block, 6, res_block_in_channels=2400,  first_conv_out_channels=1200, stride=1)
        self.conv5_x = self._make_layer(block, 3, res_block_in_channels=4800, first_conv_out_channels=2400, stride=1)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(2400*4, num_classes)

    def forward(self,x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, num_res_blocks, res_block_in_channels, first_conv_out_channels, stride):
        layers = []

        identity_conv = nn.Conv2d(res_block_in_channels, first_conv_out_channels*4, kernel_size=1,stride=stride)
        layers.append(block(res_block_in_channels, first_conv_out_channels, identity_conv, stride))

        in_channels = first_conv_out_channels*4

        for i in range(num_res_blocks - 1):
            layers.append(block(in_channels, first_conv_out_channels, identity_conv=None, stride=1))

        return nn.Sequential(*layers)
    
    