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

from vgg import AdaptedVGGNet, count_parameters
from test import test_vgg_model

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import numpy as np

os.makedirs("problem2_results", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 0.01

mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)

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

train_dataset, val_dataset = random_split(
    full_train_dataset,
    [45000, 5000],
    generator=torch.Generator().manual_seed(SEED)
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

classes = test_dataset.classes

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


vgg_experiments = {
    "vgg_baseline": {
        "dropout_rate": 0.0,
        "use_batch_norm": False
    },
    "vgg_dropout_03": {
        "dropout_rate": 0.3,
        "use_batch_norm": False
    },
    "vgg_dropout_05": {
        "dropout_rate": 0.5,
        "use_batch_norm": False
    }
}

vgg_results = {}

for model_name, settings in vgg_experiments.items():
    print("\n====================================")
    print("Training:", model_name)
    print("Dropout rate:", settings["dropout_rate"])
    print("BatchNorm:", settings["use_batch_norm"])
    print("====================================")

    model, train_losses, val_losses, val_accuracies, time_per_epoch = train_vgg_model(
        dropout_rate=settings["dropout_rate"],
        model_name=model_name,
        use_batch_norm=settings["use_batch_norm"]
    )

    test_accuracy, cm = test_vgg_model(model, model_name, test_loader, device, classes)

    torch.save(model.state_dict(), f"problem2_results/{model_name}_model.pth")

    vgg_results[model_name] = {
        "model": model,
        "dropout_rate": settings["dropout_rate"],
        "use_batch_norm": settings["use_batch_norm"],
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "test_accuracy": test_accuracy,
        "parameters": count_parameters(model),
        "time_per_epoch": time_per_epoch
    }

    print(f"{model_name} Test Accuracy: {test_accuracy:.2f}%")
    print(f"{model_name} Time per Epoch: {time_per_epoch:.2f} seconds")

print("\n====================================")
print("Training VGG with Batch Normalization Bonus")
print("====================================")

bn_model_name = "vgg_batchnorm_bonus"

bn_model, bn_train_losses, bn_val_losses, bn_val_accuracies, bn_time_per_epoch = train_vgg_model(
    dropout_rate=0.0,
    model_name=bn_model_name,
    use_batch_norm=True
)

bn_test_accuracy, bn_cm = test_vgg_model(
    bn_model,
    bn_model_name,
    test_loader,
    device,
    classes
)
torch.save(bn_model.state_dict(), f"problem2_results/{bn_model_name}_model.pth")

vgg_results[bn_model_name] = {
    "model": bn_model,
    "dropout_rate": 0.0,
    "use_batch_norm": True,
    "train_losses": bn_train_losses,
    "val_losses": bn_val_losses,
    "val_accuracies": bn_val_accuracies,
    "test_accuracy": bn_test_accuracy,
    "parameters": count_parameters(bn_model),
    "time_per_epoch": bn_time_per_epoch
}

print(f"{bn_model_name} Test Accuracy: {bn_test_accuracy:.2f}%")
print(f"{bn_model_name} Time per Epoch: {bn_time_per_epoch:.2f} seconds")

# VGG loss comparison
plt.figure(figsize=(10, 6))

for model_name in vgg_results:
    plt.plot(vgg_results[model_name]["train_losses"], label=f"{model_name} train loss")
    plt.plot(vgg_results[model_name]["val_losses"], label=f"{model_name} val loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Adapted VGGNet Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig("problem2_results/vgg_loss_comparison.png")
plt.show()


# VGG validation accuracy comparison
plt.figure(figsize=(10, 6))

for model_name in vgg_results:
    plt.plot(vgg_results[model_name]["val_accuracies"], label=f"{model_name} val accuracy")

plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy (%)")
plt.title("Adapted VGGNet Validation Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("problem2_results/vgg_validation_accuracy_comparison.png")
plt.show()

print("Final Problem 2 VGG Results")
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

print("\nVGG Model Results:")
print("--------------------------------------------------------------------------")
print("Model\t\t\tDropout\tBatchNorm\tParams\t\tTest Acc\tTime/Epoch")
print("--------------------------------------------------------------------------")

for model_name, data in vgg_results.items():
    print(
        f"{model_name}\t"
        f"{data['dropout_rate']}\t"
        f"{data['use_batch_norm']}\t\t"
        f"{data['parameters']}\t"
        f"{data['test_accuracy']:.2f}%\t\t"
        f"{data['time_per_epoch']:.2f}s"
    )

print("AlexNet vs VGGNet Comparison")
print("---------------------------------------------------------------")
print("Model\t\t\tParams\t\tTest Acc\tTime/Epoch")
print("---------------------------------------------------------------")

# Best AlexNet from Problem 1 report results
best_alexnet_name = "baseline"
best_alexnet = {
    "parameters": 1814666,
    "test_accuracy": 83.99,
    "time_per_epoch": 19.94
}

# Best VGG from Problem 2
best_vgg_name = max(vgg_results, key=lambda name: vgg_results[name]["test_accuracy"])
best_vgg = vgg_results[best_vgg_name]

print(
    f"Best AlexNet: {best_alexnet_name}\t"
    f"{best_alexnet['parameters']}\t"
    f"{best_alexnet['test_accuracy']:.2f}%\t\t"
    f"{best_alexnet['time_per_epoch']:.2f}s"
)

print(
    f"Best VGGNet: {best_vgg_name}\t"
    f"{best_vgg['parameters']}\t"
    f"{best_vgg['test_accuracy']:.2f}%\t\t"
    f"{best_vgg['time_per_epoch']:.2f}s"
)

# Bar chart comparing best AlexNet and best VGGNet
model_names = [f"AlexNet\n{best_alexnet_name}", f"VGGNet\n{best_vgg_name}"]
test_accuracies = [best_alexnet["test_accuracy"], best_vgg["test_accuracy"]]

plt.figure(figsize=(7, 5))
plt.bar(model_names, test_accuracies)
plt.ylabel("Test Accuracy (%)")
plt.title("Best AlexNet vs Best VGGNet Test Accuracy")
plt.ylim(0, 100)
plt.grid(axis="y")
plt.savefig("problem2_results/alexnet_vs_vgg_test_accuracy.png")
plt.show()
