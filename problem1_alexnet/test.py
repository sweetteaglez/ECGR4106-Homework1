import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def test_model(model, model_name):
    model.eval()

    correct = 0
    total = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_accuracy = 100 * correct / total

    cm = confusion_matrix(all_labels, all_predictions)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=classes
    )

    disp.plot(xticks_rotation=45)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(f"problem1_results/{model_name}_confusion_matrix.png")
    plt.show()

    return test_accuracy, cm

###
def visualize_first_layer_filters(model):
    first_conv_layer = model.features[0]
    filters = first_conv_layer.weight.data.cpu()

    # Normalize filters for display
    filters = (filters - filters.min()) / (filters.max() - filters.min())

    plt.figure(figsize=(8, 8))

    for i in range(16):
        plt.subplot(4, 4, i + 1)
        img = filters[i].permute(1, 2, 0)
        plt.imshow(img)
        plt.axis("off")

    plt.suptitle("First Convolutional Layer Filters - Baseline AlexNet")
    plt.tight_layout()
    plt.savefig("problem1_results/first_layer_filters.png")
    plt.show()


visualize_first_layer_filters(results["baseline"]["model"])
