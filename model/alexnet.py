from model.stochastic_grad_pruning import *


# adapt from torch.models.alexnet
class AlexNet_SFA(nn.Module):
    def __init__(self, prune_rate, train_mode_conv, train_mode_fc, angle_measurement):
        super(AlexNet_SFA, self).__init__()

        Conv2d = StochasticGradPruneConv2d
        Linear = StochasticGradPruneLinear

        self.features = nn.Sequential(
            Conv2d(3, 64, kernel_size=11, stride=4, padding=2,
                   bias=False, train_mode=train_mode_conv, angle_mea=angle_measurement, prune_rate=prune_rate),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            Conv2d(64, 192, kernel_size=5, padding=2,
                   bias=False, train_mode=train_mode_conv, angle_mea=angle_measurement, prune_rate=prune_rate),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            Conv2d(192, 384, kernel_size=3, padding=1,
                   bias=False, train_mode=train_mode_conv, angle_mea=angle_measurement, prune_rate=prune_rate),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            Conv2d(384, 256, kernel_size=3, padding=1,
                   bias=False, train_mode=train_mode_conv, angle_mea=angle_measurement, prune_rate=prune_rate),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            Conv2d(256, 256, kernel_size=3, padding=1,
                   bias=False, train_mode=train_mode_conv, angle_mea=angle_measurement, prune_rate=prune_rate),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            Linear(256 * 6 * 6, 4096, bias=False,
                   train_mode=train_mode_fc, angle_mea=angle_measurement, prune_rate=prune_rate),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            Linear(4096, 4096, bias=False,
                   train_mode=train_mode_fc, angle_mea=angle_measurement, prune_rate=prune_rate),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            Linear(4096, 1000, bias=False,
                   train_mode=train_mode_fc, angle_mea=angle_measurement, prune_rate=prune_rate),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
