##
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter


class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()

        self.bottleneck = False if in_channels == out_channels else True

        f_lst = OrderedDict()

        if not self.bottleneck:
            f_lst['conv1'] = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                   kernel_size=3, stride=1, padding=1, bias=True)
        else:
            f_lst['conv1'] = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                       kernel_size=3, stride=2, padding=1, bias=True)
            self.add_ch = out_channels - in_channels
        f_lst['bn1'] = nn.BatchNorm2d(num_features=out_channels)

        f_lst['conv2'] = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=3, stride=1, padding=1, bias=True)
        f_lst['bn2'] = nn.BatchNorm2d(num_features=out_channels)

        self.block = nn.Sequential(f_lst)

    def zero_pad(self, x):
        x = F.max_pool2d(x, kernel_size=1, stride=2)
        x = F.pad(x, (0,0,0,0,0,self.add_ch))

        return x

    def forward(self, x):
        y = self.block(x)

        if not self.bottleneck:
            output = x + y
        else:
            output = self.zero_pad(x) + y
        output = F.relu(output)

        return output



class ResNet(nn.Module):
    def __init__(self, n):
        super(ResNet, self).__init__()

        conv1_1_lst = OrderedDict()
        conv1_1_lst['conv1_1'] = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True)
        conv1_1_lst['bn1_1'] = nn.BatchNorm2d(num_features=16)

        self.conv1_1 = nn.Sequential(conv1_1_lst)

        layer_lst = OrderedDict()

        for i in range(1, n+1):
            layer_lst['block_1_' + str(i)] = Block(16, 16)

        for i in range(1, n+1):
            if i == 1:
                layer_lst['block_2_' + str(i)] = Block(16, 32)
            else:
                layer_lst['block_2_' + str(i)] = Block(32, 32)

        for i in range(1, n+1):
            if i == 1:
                layer_lst['block_3_' + str(i)] = Block(32, 64)
            else:
                layer_lst['block_3_' + str(i)] = Block(64, 64)

        self.layer = nn.Sequential(layer_lst)

        self.global_average_pool = nn.AvgPool2d(kernel_size=8, stride=1 )

        self.fc = nn.Linear(in_features=64, out_features=10, bias=True)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))

        x = self.layer(x)

        x = self.global_average_pool(x)
        x = x.view(-1, 64)

        x = self.fc(x)

        return x



if __name__ == '__main__':
    net = ResNet(18)
    input = torch.empty(3, 3, 32, 32)
    output = net(input)
    print(output.shape)

    # block = ResNet(18)
    # writer = SummaryWriter('runs')
    # writer.add_graph(block, torch.empty(10, 3, 32, 32))
    # writer.close()

    # layer1_lst = OrderedDict()
    #
    # for i in range(1, 18):
    #     layer1_lst['block_' + str(i)] = Block(16, 16)
    # print(layer1_lst)

##

