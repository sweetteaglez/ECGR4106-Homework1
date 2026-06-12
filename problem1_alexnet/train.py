# imports first

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, random_split
from torchsummary import summary

import matplotlib.pyplot as plt
import numpy as np
import os
import time

# ECGR 4106 Homework 1 - Problem 1
# Name: Samantha Gonzalez
# Modified AlexNet on CIFAR-10

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

os.makedirs("problem1_results", exist_ok=True)

BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 0.01

# CIFAR-10 normalization values
mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)

# Same augmentation should be used for all models in the homework
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

full_train_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=train_transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=test_transform
)

# Consistent 45,000 train / 5,000 validation split
train_dataset, val_dataset = random_split(
    full_train_dataset,
    [45000, 5000],
    generator=torch.Generator().manual_seed(SEED)
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

classes = test_dataset.classes

print("Train size:", len(train_dataset))
print("Validation size:", len(val_dataset))
print("Test size:", len(test_dataset))
print("Classes:", classes)

baseline_model = ModifiedAlexNet(dropout_rate=0.0).to(device)

print("Modified AlexNet parameter count:", count_parameters(baseline_model))
print("Original AlexNet parameter count: about 61,000,000")

summary(baseline_model, (3, 32, 32))
