import torch
import torch.nn as nn

class AdaptedVGGNet(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.0, use_batch_norm=False):
        super(AdaptedVGGNet, self).__init__()

        # This is based on VGG-11 style structure.
        # I reduced the number of channels so the parameter count is closer
        # to my modified AlexNet from Problem 1.
        self.use_batch_norm = use_batch_norm

        self.features = nn.Sequential(
            self.vgg_block(3, 32, use_batch_norm),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 32 -> 16

            self.vgg_block(32, 64, use_batch_norm),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 16 -> 8

            self.vgg_block(64, 128, use_batch_norm),
            self.vgg_block(128, 128, use_batch_norm),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 8 -> 4

            self.vgg_block(128, 192, use_batch_norm),
            self.vgg_block(192, 192, use_batch_norm),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 4 -> 2

            self.vgg_block(192, 192, use_batch_norm),
            self.vgg_block(192, 192, use_batch_norm),
            nn.MaxPool2d(kernel_size=2, stride=2)    # 2 -> 1
        )

        # AdaptiveAvgPool keeps the classifier simple and works well for CIFAR-10.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Smaller fully connected layers than original VGG.
        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(192, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(128, num_classes)
        )

    def vgg_block(self, in_channels, out_channels, use_batch_norm):
        if use_batch_norm:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU()
            )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
