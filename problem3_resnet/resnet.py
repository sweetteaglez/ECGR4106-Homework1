import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        # First 3x3 convolution
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        # Second 3x3 convolution
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut path.
        # If the input and output sizes are different, use 1x1 convolution.
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Skip connection: output = F(x) + x
        out = out + identity
        out = self.relu(out)

        return out


class CIFARResNet(nn.Module):
    def __init__(self, block_counts, num_classes=10, dropout_rate=0.0):
        super(CIFARResNet, self).__init__()

        self.in_channels = 64
        self.dropout_rate = dropout_rate

        # CIFAR-10 version:
        # Use 3x3 convolution instead of ImageNet 7x7 convolution.
        # Do not use the first max pool because CIFAR-10 images are small.
        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        # Four residual stages
        self.layer1 = self.make_layer(64, block_counts[0], stride=1)
        self.layer2 = self.make_layer(128, block_counts[1], stride=2)
        self.layer3 = self.make_layer(256, block_counts[2], stride=2)
        self.layer4 = self.make_layer(512, block_counts[3], stride=2)

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Dropout is applied after global average pooling
        self.dropout = nn.Dropout(p=dropout_rate)

        # Final classifier
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, out_channels, num_blocks, stride):
        layers = []

        # First block may downsample
        layers.append(BasicBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels

        # Remaining blocks keep same size
        for i in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)

        out = self.dropout(out)

        out = self.fc(out)

        return out


def ResNet11(dropout_rate=0.0):
    # Smaller ResNet baseline.
    # This has fewer residual blocks than ResNet-18.
    return CIFARResNet(
        block_counts=[1, 1, 1, 1],
        dropout_rate=dropout_rate
    )


def ResNet18(dropout_rate=0.0):
    # ResNet-18 has four stages with two BasicBlocks each.
    return CIFARResNet(
        block_counts=[2, 2, 2, 2],
        dropout_rate=dropout_rate
    )

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
