# ECGR 4106 Homework 1 - Problem 3
# Name: Samantha Gonzalez
# ResNet-11 vs ResNet-18 on CIFAR-10

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

os.makedirs("problem3_results", exist_ok=True)

RESNET_EPOCHS = 50

print("Starting Problem 3: ResNet-11 vs ResNet-18")
print("Using device:", device)

resnet11_test = ResNet11().to(device)
resnet18_test = ResNet18().to(device)

print("ResNet-11 parameter count:", count_parameters(resnet11_test))
print("ResNet-18 parameter count:", count_parameters(resnet18_test))
