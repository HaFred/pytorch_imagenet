from model.stochastic_grad_pruning import *
import torch.nn.functional as F
import torch
from torch.utils.checkpoint import checkpoint

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, conv_act, train_mode,
                 prune_rate, prune_flag, angle_measurement, stride=1):
        super(BasicBlock, self).__init__()
        if prune_flag == 'PureFA':  # the BP will be done in the FA_wrapper
            Conv2d = PureFAConv2d
            Linear = PureFALinear
        elif prune_flag == 'StochasticFA':
            Conv2d = StochasticGradPruneConv2d
            Linear = StochasticGradPruneLinear
            # nn.BatchNorm2d = GradPruneBN
        else:
            raise ValueError("=== ERROR: the {} is not supported".format(prune_flag))
        self.conv_act = conv_act
        self.conv1 = Conv2d(in_planes, planes, kernel_size=3, train_mode=train_mode,
                            prune_rate=prune_rate, angle_mea=angle_measurement,  # the angle mea here could turn on to visualize the grad
                            # angle of EfficientGrad
                            stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, train_mode=train_mode,
                            prune_rate=prune_rate, angle_mea=angle_measurement,
                            stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                Conv2d(in_planes, self.expansion * planes, train_mode=train_mode, angle_mea=angle_measurement,
                       prune_rate=prune_rate, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        if self.conv_act == 'relu':
            out = checkpoint(torch.relu(self.bn1(self.conv1(x))))
            out = checkpoint(self.bn2(self.conv2(out)))
            out += checkpoint(self.shortcut(x))
            out = checkpoint(torch.relu(out))
        elif self.conv_act == 'tanh':
            out = torch.tanh(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = torch.tanh(out)
        else:
            raise ValueError('Only relu and tanh supported')
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks,
                 conv_act,
                 train_mode_conv, train_mode_fc,
                 prune_flag, prune_percent, angle_measurement, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv_act = conv_act
        if prune_flag == 'PureFA':  # the BP will be done in the FA_wrapper
            Conv2d = PureFAConv2d
            Linear = PureFALinear
        elif prune_flag == 'StochasticFA':
            Conv2d = StochasticGradPruneConv2d
            Linear = StochasticGradPruneLinear
            # nn.BatchNorm2d = GradPruneBN
        else:
            raise ValueError("=== ERROR: the {} is not supported".format(prune_flag))
        # layer0 has error_grad, but since image_grad is not needed, the layer0 cannot get in the hook in the
        # FA_wrapper, thus no angle measured
        self.layer0 = nn.Sequential(
            Conv2d(in_channels=3, out_channels=64, kernel_size=3, train_mode=train_mode_conv,
                   angle_mea=angle_measurement,
                   # for the top layer, no error_grad needs to be cal, thus here won't have fb
                   prune_rate=prune_percent, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64)  # the grad is in the PureFAConv2d, but not fa_wrapper
        )
        self.layer1 = self._make_layer(block, 64, num_blocks[0], conv_act=conv_act,
                                       stride=1, prune_flag=prune_flag,
                                       train_mode=train_mode_conv, prune_rate=prune_percent,
                                       angle_measurement=angle_measurement)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], conv_act=conv_act,
                                       stride=2, prune_flag=prune_flag,
                                       train_mode=train_mode_conv, prune_rate=prune_percent,
                                       angle_measurement=angle_measurement)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], conv_act=conv_act,
                                       stride=2, prune_flag=prune_flag,
                                       train_mode=train_mode_conv, prune_rate=prune_percent,
                                       angle_measurement=angle_measurement)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], conv_act=conv_act,
                                       stride=2, prune_flag=prune_flag,
                                       train_mode=train_mode_conv, prune_rate=prune_percent,
                                       angle_measurement=angle_measurement)
        self.linear = nn.Sequential(Linear(512 * block.expansion, num_classes, train_mode=train_mode_fc,
                                           angle_mea=angle_measurement,
                                           prune_rate=prune_percent))

    def _make_layer(self, block, planes, num_blocks, conv_act, stride, prune_flag, train_mode, prune_rate, angle_measurement):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, conv_act=conv_act,
                                stride=stride, prune_flag=prune_flag,
                                train_mode=train_mode,
                                prune_rate=prune_rate, angle_measurement=angle_measurement))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.conv_act == 'relu':
            out = checkpoint(torch.relu(self.layer0(x)))
            out = checkpoint(self.layer1(out))
            out = checkpoint(self.layer2(out))
            out = checkpoint(self.layer3(out))
            out = checkpoint(self.layer4(out))
            out = checkpoint(F.avg_pool2d(out, 4))
            out = checkpoint(out.view(out.size(0), -1))
            out = checkpoint(self.linear(out))
        elif self.conv_act == 'tanh':
            out = torch.tanh(self.layer0(x))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
        else:
            raise ValueError('Only relu and tanh supported')
        return out


def ResNet18(conv_act,
             train_mode_conv, train_mode_fc,
             prune_flag, prune_percent, angle_measurement):
    return ResNet(BasicBlock, [2, 2, 2, 2],
                  conv_act,
                  train_mode_conv, train_mode_fc,
                  prune_flag, prune_percent, angle_measurement)