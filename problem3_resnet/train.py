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


def train_resnet_model(model_type, dropout_rate, model_name):
    if model_type == "resnet11":
        model = ResNet11(dropout_rate=dropout_rate).to(device)
    elif model_type == "resnet18":
        model = ResNet18(dropout_rate=dropout_rate).to(device)
    else:
        raise ValueError("model_type must be 'resnet11' or 'resnet18'")

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=0.9
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=20,
        gamma=0.1
    )

    train_losses = []
    val_losses = []
    val_accuracies = []

    start_time = time.time()

    for epoch in range(RESNET_EPOCHS):
        model.train()
        running_train_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        running_val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = running_val_loss / len(val_loader)
        val_accuracy = 100 * correct / total

        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        scheduler.step()

        print(
            f"{model_name} | Epoch {epoch+1:02d}/{RESNET_EPOCHS} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.2f}%"
        )

    total_time = time.time() - start_time
    time_per_epoch = total_time / RESNET_EPOCHS

    return model, train_losses, val_losses, val_accuracies, time_per_epoch
