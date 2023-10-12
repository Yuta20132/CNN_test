import torch 
import torch.nn as nn
from torchinfo import summary

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size = 3,
        stride = stride,
        padding = 1,
        bias = False,
    )

def conv1x1(in_channels,out_channels, stride=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=stride,
        bias=False
    )


class BasicBlock(nn.Module):

    """
    1. Conv2D (kernel_size=3, padding=1, stride=1 or 2)
    2. BatchNorm2d
    3. ReLU
    4. Conv2D (kernel_size=3, padding=1, stride=1)
    5. BatchNorm2d
    6. 形状が入力と異なる場合は、1×1 の畳み込み層で線形変換を行う
    7. shortcut connection と結合します。
    8. ReLU
    
    
    """
    expansion = 1

    def __init__(
            self,
            in_channels,
            channels,
            stride = 1
    ):
        super().__init__()
        self.conv1 = conv3x3(in_channels,channels,stride)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(channels,channels)
        self.bn2 = nn.BatchNorm2d(channels)


        # 入力と出力のチャンネル数が異なる場合，xをダウンサンプリング
        if in_channels != channels * self.expansion:
            self.shortcut = nn.Sequential(
                conv1x1(in_channels, channels * self.expansion, stride),

            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)

        out = self.relu(out)

        return out
    
class Bottleneck(nn.Module):
    expansion = 4  # 出力のチャンネル数を入力のチャンネル数の何倍に拡大するか

    def __init__(self, in_channels, channels, stride=1):
        super().__init__()
        self.conv1 = conv1x1(in_channels, channels)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = conv3x3(channels, channels, stride)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = conv1x1(channels, channels * self.expansion)
        self.bn3 = nn.BatchNorm2d(channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        # 入力と出力のチャンネル数が異なる場合、x をダウンサンプリングする。
        if in_channels != channels * self.expansion:
            self.shortcut = nn.Sequential(
                conv1x1(in_channels, channels * self.expansion, stride),
                nn.BatchNorm2d(channels * self.expansion),
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += self.shortcut(x)

        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block,layers,num_classes = 1000):
        super().__init__()

        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            3, self.in_channels, kernel_size = 7,stride = 2, padding=3,bias=False
        )


        """
        1. Conv2D (out_channels=64, kernel_size=7, padding=2, stride=3)
        2. BatchNorm2d
        3. MaxPool2d (kernel_size=3, stride=2, padding=1)
        4. Residual Blocks (in_channels=64)
        5. Residual Blocks (in_channels=128)
        6. Residual Blocks (in_channels=256)
        7. Residual Blocks (in_channels=512)
        8. Global Average Pooling
        9. Linear (out_channels=num_classes)
        """

        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding = 1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride = 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        #重みを初期化
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)

    
    def _make_layer(self, block, channels, blocks, stride):

        layers = []

        #最初のResidual Block
        layers.append(block(self.in_channels, channels, stride))

        #残りのResidual Block
        self.in_channels = channels * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, channels))


        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


#ResNetを構築

def resnet18():
    return ResNet(BasicBlock,[2,2,2,2])


def resnet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def resnet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def resnet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def resnet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])




model = resnet18()

summary(
    model,
    input_size=(1, 3, 224, 224),
    col_names=["output_size", "num_params"],
)
