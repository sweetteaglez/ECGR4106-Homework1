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


def train_vgg_model(dropout_rate, model_name, use_batch_norm=False):
    model = AdaptedVGGNet(
        dropout_rate=dropout_rate,
        use_batch_norm=use_batch_norm
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=0.9
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=15,
        gamma=0.1
    )

    train_losses = []
    val_losses = []
    val_accuracies = []

    start_time = time.time()

    for epoch in range(EPOCHS):
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

        # Validation step
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
            f"{model_name} | Epoch {epoch+1:02d}/{EPOCHS} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.2f}%"
        )

    total_time = time.time() - start_time
    time_per_epoch = total_time / EPOCHS

    return model, train_losses, val_losses, val_accuracies, time_per_epoch
