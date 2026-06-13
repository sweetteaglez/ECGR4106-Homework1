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
from test import test_resnet_model

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

plt.figure(figsize=(12, 7))

for model_name in resnet_results:
    plt.plot(
        resnet_results[model_name]["train_losses"],
        label=f"{model_name} train loss"
    )
    plt.plot(
        resnet_results[model_name]["val_losses"],
        label=f"{model_name} val loss"
    )

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("ResNet Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig("problem3_results/resnet_loss_comparison.png")
plt.show()

plt.figure(figsize=(12, 7))

for model_name in resnet_results:
    plt.plot(
        resnet_results[model_name]["val_accuracies"],
        label=f"{model_name} val accuracy"
    )

plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy (%)")
plt.title("ResNet Validation Accuracy Comparison")
plt.legend()
plt.grid(True)
plt.savefig("problem3_results/resnet_validation_accuracy_comparison.png")
plt.show()


print("Final Problem 3 ResNet Results")
print("Random Seed:", SEED)
print("Batch Size:", BATCH_SIZE)
print("Epochs:", RESNET_EPOCHS)
print("Learning Rate:", LEARNING_RATE)
print("Optimizer: SGD with momentum = 0.9")
print("Scheduler: StepLR, step_size = 20, gamma = 0.1")
print("Device:", device)

# Best ResNet-11
resnet11_names = [
    name for name in resnet_results
    if resnet_results[name]["model_type"] == "resnet11"
]

best_resnet11_name = max(
    resnet11_names,
    key=lambda name: resnet_results[name]["test_accuracy"]
)

best_resnet11 = resnet_results[best_resnet11_name]


# Best ResNet-18
resnet18_names = [
    name for name in resnet_results
    if resnet_results[name]["model_type"] == "resnet18"
]

best_resnet18_name = max(
    resnet18_names,
    key=lambda name: resnet_results[name]["test_accuracy"]
)

best_resnet18 = resnet_results[best_resnet18_name]


print("Best ResNet-11 vs Best ResNet-18")
print("---------------------------------------------------------------")
print("Model\t\tParams\t\tTest Acc\tTime/Epoch")
print("---------------------------------------------------------------")

print(
    f"{best_resnet11_name}\t"
    f"{best_resnet11['parameters']}\t"
    f"{best_resnet11['test_accuracy']:.2f}%\t\t"
    f"{best_resnet11['time_per_epoch']:.2f}s"
)

print(
    f"{best_resnet18_name}\t"
    f"{best_resnet18['parameters']}\t"
    f"{best_resnet18['test_accuracy']:.2f}%\t\t"
    f"{best_resnet18['time_per_epoch']:.2f}s"
)


plt.figure(figsize=(7, 5))

model_names = [
    f"ResNet-11\n{best_resnet11_name}",
    f"ResNet-18\n{best_resnet18_name}"
]

test_accuracies = [
    best_resnet11["test_accuracy"],
    best_resnet18["test_accuracy"]
]

plt.bar(model_names, test_accuracies)
plt.ylabel("Test Accuracy (%)")
plt.title("Best ResNet-11 vs Best ResNet-18 Test Accuracy")
plt.ylim(0, 100)
plt.grid(axis="y")
plt.savefig("problem3_results/resnet11_vs_resnet18_accuracy.png")
plt.show()

# Best AlexNet from Problem 1
best_alexnet_name = max(
    results,
    key=lambda name: results[name]["test_accuracy"]
)
best_alexnet = results[best_alexnet_name]

# Best VGGNet from Problem 2
best_vgg_name = max(
    vgg_results,
    key=lambda name: vgg_results[name]["test_accuracy"]
)
best_vgg = vgg_results[best_vgg_name]

# Best ResNet-11
best_resnet11_name = max(
    resnet11_names,
    key=lambda name: resnet_results[name]["test_accuracy"]
)
best_resnet11 = resnet_results[best_resnet11_name]

# Best ResNet-18
best_resnet18_name = max(
    resnet18_names,
    key=lambda name: resnet_results[name]["test_accuracy"]
)
best_resnet18 = resnet_results[best_resnet18_name]


final_models = {
    f"AlexNet\n{best_alexnet_name}": best_alexnet,
    f"VGGNet\n{best_vgg_name}": best_vgg,
    f"ResNet-11\n{best_resnet11_name}": best_resnet11,
    f"ResNet-18\n{best_resnet18_name}": best_resnet18
}


print("Final Comparison Across All Four Architectures")
print("---------------------------------------------------------------------")
print("Model\t\t\tParams\t\tTest Acc\tTime/Epoch")
print("---------------------------------------------------------------------")

for model_name, data in final_models.items():
    clean_name = model_name.replace("\n", " ")
    print(
        f"{clean_name}\t"
        f"{data['parameters']}\t"
        f"{data['test_accuracy']:.2f}%\t\t"
        f"{data['time_per_epoch']:.2f}s"
    )


plt.figure(figsize=(10, 6))

names = list(final_models.keys())
accuracies = [final_models[name]["test_accuracy"] for name in names]

plt.bar(names, accuracies)
plt.ylabel("Test Accuracy (%)")
plt.title("Final Test Accuracy Comparison Across Models")
plt.ylim(0, 100)
plt.grid(axis="y")
plt.savefig("problem3_results/final_all_models_accuracy_comparison.png")
plt.show()

# Best AlexNet from Problem 1
best_alexnet_name = max(
    results,
    key=lambda name: results[name]["test_accuracy"]
)
best_alexnet = results[best_alexnet_name]

# Best VGGNet from Problem 2
best_vgg_name = max(
    vgg_results,
    key=lambda name: vgg_results[name]["test_accuracy"]
)
best_vgg = vgg_results[best_vgg_name]

# Best ResNet-11
best_resnet11_name = max(
    resnet11_names,
    key=lambda name: resnet_results[name]["test_accuracy"]
)
best_resnet11 = resnet_results[best_resnet11_name]

# Best ResNet-18
best_resnet18_name = max(
    resnet18_names,
    key=lambda name: resnet_results[name]["test_accuracy"]
)
best_resnet18 = resnet_results[best_resnet18_name]


final_models = {
    f"AlexNet\n{best_alexnet_name}": best_alexnet,
    f"VGGNet\n{best_vgg_name}": best_vgg,
    f"ResNet-11\n{best_resnet11_name}": best_resnet11,
    f"ResNet-18\n{best_resnet18_name}": best_resnet18
}


print("Final Comparison Across All Four Architectures")
print("---------------------------------------------------------------------")
print("Model\t\t\tParams\t\tTest Acc\tTime/Epoch")
print("---------------------------------------------------------------------")

for model_name, data in final_models.items():
    clean_name = model_name.replace("\n", " ")
    print(
        f"{clean_name}\t"
        f"{data['parameters']}\t"
        f"{data['test_accuracy']:.2f}%\t\t"
        f"{data['time_per_epoch']:.2f}s"
    )


plt.figure(figsize=(10, 6))

names = list(final_models.keys())
accuracies = [final_models[name]["test_accuracy"] for name in names]

plt.bar(names, accuracies)
plt.ylabel("Test Accuracy (%)")
plt.title("Final Test Accuracy Comparison Across Models")
plt.ylim(0, 100)
plt.grid(axis="y")
plt.savefig("problem3_results/final_all_models_accuracy_comparison.png")
plt.show()

if torch.cuda.is_available():
    print("Hardware:", torch.cuda.get_device_name(0))
else:
    print("Hardware: CPU")

print("\nResNet Results:")
print("----------------------------------------------------------------------------")
print("Model\t\t\tType\t\tDropout\tParams\t\tTest Acc\tTime/Epoch")
print("----------------------------------------------------------------------------")

for model_name, data in resnet_results.items():
    print(
        f"{model_name}\t"
        f"{data['model_type']}\t"
        f"{data['dropout_rate']}\t"
        f"{data['parameters']}\t"
        f"{data['test_accuracy']:.2f}%\t\t"
        f"{data['time_per_epoch']:.2f}s"
    )
        
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

test_accuracy, cm = test_resnet_model(
    model,
    model_name,
    test_loader,
    device,
    classes
)

resnet_experiments = {
    "resnet11_baseline": {
        "model_type": "resnet11",
        "dropout_rate": 0.0
    },
    "resnet11_dropout_03": {
        "model_type": "resnet11",
        "dropout_rate": 0.3
    },
    "resnet11_dropout_05": {
        "model_type": "resnet11",
        "dropout_rate": 0.5
    },
    "resnet18_baseline": {
        "model_type": "resnet18",
        "dropout_rate": 0.0
    },
    "resnet18_dropout_03": {
        "model_type": "resnet18",
        "dropout_rate": 0.3
    },
    "resnet18_dropout_05": {
        "model_type": "resnet18",
        "dropout_rate": 0.5
    }
}

resnet_results = {}

for model_name, settings in resnet_experiments.items():
    print("\n====================================")
    print("Training:", model_name)
    print("Model type:", settings["model_type"])
    print("Dropout rate:", settings["dropout_rate"])
    print("====================================")

    model, train_losses, val_losses, val_accuracies, time_per_epoch = train_resnet_model(
        model_type=settings["model_type"],
        dropout_rate=settings["dropout_rate"],
        model_name=model_name
    )

    test_accuracy, cm = test_resnet_model(model, model_name)

    torch.save(model.state_dict(), f"problem3_results/{model_name}_model.pth")

    resnet_results[model_name] = {
        "model": model,
        "model_type": settings["model_type"],
        "dropout_rate": settings["dropout_rate"],
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "test_accuracy": test_accuracy,
        "parameters": count_parameters(model),
        "time_per_epoch": time_per_epoch
    }

    print(f"{model_name} Test Accuracy: {test_accuracy:.2f}%")
    print(f"{model_name} Time per Epoch: {time_per_epoch:.2f} seconds")
