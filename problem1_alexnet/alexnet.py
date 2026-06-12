import torch
import torch.nn as nn

class ModifiedAlexNet(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.0):
        super(ModifiedAlexNet, self).__init__()

        # CIFAR-10 images are only 32x32.
        # I use smaller 3x3 filters instead of large AlexNet filters.
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32 -> 16x16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16 -> 8x8

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)   # 8x8 -> 4x4
        )

        # Final feature map is 128 x 4 x 4.
        # Fully connected layers are much smaller than original AlexNet.
        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

