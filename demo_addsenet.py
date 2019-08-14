from collections import OrderedDict
import torch.nn.functional as F
import torch
from torch import nn
from torch.autograd import Variable


#  这里还是不懂它的机制,懂了
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()

        # 返回1X1大小的特征图，通道数不变
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()

        # 全局平均池化，batch和channel和原来一样保持不变
        y = self.avg_pool(x).view(b, c)

        # 全连接层+池化
        y = self.fc(y).view(b, c, 1, 1)
        a = x * y.expand_as(x)

        # 和原特征图相乘
        return a


"""
       num_input_features:输入特征图个数
       growth_rate: 增长速率，第二个卷积层输出特征图
       grow_rate * bn_size: 第一个卷积层输出特征图
       drop_rate: dropout失活率
"""


class _DenseLayer(nn.Sequential):  # 卷积块：BN->ReLU->1x1Conv->BN->ReLU->3x3Conv
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate
        self.se = SELayer(growth_rate)

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        # 先进行forward然后进行se，再之后dropout_rate(这种策略会得到更好的特征层)
        # 那其实我现在的想法是，你se_net在干的事不就是在选择哪些是有用的东西嘛
        # 我再加入正则项，去训练，如果把有效的信息给抛弃了不就没用了嘛
        # 目前我给出的回答是：加入dropout是为了更能泛化
        # 1）我先把dropout屏蔽掉试一下，精度提高了2%（在reduction=8时，train为9000的时候）
        # 2）dropout不屏蔽，数据量为18000的时候看一下,精度变低了71.多
        # 3）dropout屏蔽,数据量改为18000，
        return torch.cat([x, new_features], 1)
        """
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)
        """

"""
num_input_features:输入特征图个数
num_output_features:输出特征图个数，为num_input_features//2
"""


class _Transition(nn.Sequential):  # 过渡层，将特征图个数减半
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        "num_layers:每个block内dense layer层数"
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class DenseNet(nn.Module):  # 121层DenseNet
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0.5, num_classes=3):

        super(DenseNet, self).__init__()
        # 第一个卷积层
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # 每个denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('dense_block:%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:  # 每两个dense block之间增加一个过渡层
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition:%d' % (i + 1), trans)
                num_features = num_features // 2

        #  batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # 分类器
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        features = self.features(x)
        # print("features:", features.shape)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7).view(features.size(0), -1)
        out = self.classifier(out)
        return out


"""
test_net = DenseNet()
test_x = Variable(torch.zeros(2, 3, 128, 173))
test_y = test_net(test_x)
print(test_y)
"""
