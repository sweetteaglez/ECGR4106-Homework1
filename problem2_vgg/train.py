# ECGR 4106 Homework 1 - Problem 2
# Name: Samantha Gonzalez
# Modified VGGNet on CIFAR-10

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

os.makedirs("problem2_results", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Starting Problem 2: Modified VGGNet")
print("Using device:", device)


vgg_test_model = AdaptedVGGNet(dropout_rate=0.0, use_batch_norm=False).to(device)

print("Adapted VGGNet parameter count:", count_parameters(vgg_test_model))
print("Modified AlexNet parameter count from Problem 1: about 1,814,666")
print("Original VGG models are much larger, so this version is reduced for CIFAR-10.")
