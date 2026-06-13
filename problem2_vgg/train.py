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
