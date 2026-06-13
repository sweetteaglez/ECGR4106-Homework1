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

from alexnet import ModifiedAlexNet, count_parameters
from test import test_model, visualize_first_layer_filters

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

#Train baseline and dropout models
baseline_model = ModifiedAlexNet(dropout_rate=0.0).to(device)

print("Modified AlexNet parameter count:", count_parameters(baseline_model))
print("Original AlexNet parameter count: about 61,000,000")

summary(baseline_model, (3, 32, 32))

def train_one_model(dropout_rate, model_name):
    model = ModifiedAlexNet(dropout_rate=dropout_rate).to(device)

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
            f"{model_name} | Epoch {epoch+1:02d}/{EPOCHS} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.2f}%"
        )

    total_time = time.time() - start_time
    time_per_epoch = total_time / EPOCHS

    return model, train_losses, val_losses, val_accuracies, time_per_epoch

experiments = {
    "baseline": 0.0,
    "dropout_03": 0.3,
    "dropout_05": 0.5
}

results = {}

for model_name, dropout_rate in experiments.items():
    print("\n====================================")
    print("Training:", model_name)
    print("Dropout rate:", dropout_rate)
    print("====================================")

    model, train_losses, val_losses, val_accuracies, time_per_epoch = train_one_model(
        dropout_rate,
        model_name
    )

    test_accuracy, cm = test_model(model, model_name, test_loader, device, classes)

    torch.save(model.state_dict(), f"problem1_results/{model_name}_model.pth")

    results[model_name] = {
        "model": model,
        "dropout_rate": dropout_rate,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "test_accuracy": test_accuracy,
        "parameters": count_parameters(model),
        "time_per_epoch": time_per_epoch
    }

    print(f"{model_name} Test Accuracy: {test_accuracy:.2f}%")
    print(f"{model_name} Time per Epoch: {time_per_epoch:.2f} seconds")

# Loss comparison
plt.figure(figsize=(10, 6))

for model_name in results:
    plt.plot(results[model_name]["train_losses"], label=f"{model_name} train loss")
    plt.plot(results[model_name]["val_losses"], label=f"{model_name} val loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Modified AlexNet Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig("problem1_results/alexnet_loss_comparison.png")
plt.show()


# Validation accuracy comparison
plt.figure(figsize=(10, 6))

for model_name in results:
    plt.plot(results[model_name]["val_accuracies"], label=f"{model_name} val accuracy")

plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy (%)")
plt.title("Modified AlexNet Validation Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("problem1_results/alexnet_validation_accuracy_comparison.png")
plt.show()

visualize_first_layer_filters(results["baseline"]["model"])

# Final results printout

print("Final Problem 1 Results")
print("Random Seed:", SEED)
print("Batch Size:", BATCH_SIZE)
print("Epochs:", EPOCHS)
print("Learning Rate:", LEARNING_RATE)
print("Optimizer: SGD with momentum = 0.9")
print("Scheduler: StepLR, step_size = 15, gamma = 0.1")
print("Device:", device)

if torch.cuda.is_available():
    print("Hardware:", torch.cuda.get_device_name(0))
else:
    print("Hardware: CPU")

print("\nModel Results:")
print("------------------------------------------------------------")
print("Model\t\tDropout\tParams\t\tTest Acc\tTime/Epoch")
print("------------------------------------------------------------")

for model_name, data in results.items():
    print(
        f"{model_name}\t"
        f"{data['dropout_rate']}\t"
        f"{data['parameters']}\t"
        f"{data['test_accuracy']:.2f}%\t\t"
        f"{data['time_per_epoch']:.2f}s"
    )
