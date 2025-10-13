import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay
)
import numpy as np
import os

try:
    import google.colab
    IN_COLAB = True
except:
    IN_COLAB = False

if not IN_COLAB:
    from dataset import get_dataloader
    from modules import ConvNext

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 450
LEARNING_RATE = 4e-3
BATCH_SIZE = 512
WEIGHT_DECAY = 0.01
LABEL_SMOOTHING = 0.1

def train_one_epoch(model, dataloader, criterion, optimizer, epoch):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(dataloader, desc=f"Epoch {epoch} [Train]"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += batch_size

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

def validate(model, dataloader, criterion, epoch):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.inference_mode():
        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch} [Val]"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += batch_size

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total

    # Compute additional metrics
    precision = precision_score(y_true=all_labels, y_pred=all_preds)
    recall = recall_score(y_true=all_labels, y_pred=all_preds)
    f1 = f1_score(y_true=all_labels, y_pred=all_preds)
    cm = confusion_matrix(y_true=all_labels, y_pred=all_preds)

    return avg_loss, accuracy, precision, recall, f1, cm

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, output_dir="plots"):
    """
    Plot and save training & validation loss and accuracy curves over epochs.

    Args:
        train_losses (list[float]): Training loss per epoch.
        val_losses (list[float]): Validation loss per epoch.
        train_accs (list[float]): Training accuracy per epoch.
        val_accs (list[float]): Validation accuracy per epoch.
        output_dir (str): Directory to save the plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label="Train Acc")
    plt.plot(epochs, val_accs, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training & Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved training curves to {save_path}")

def run_inference_on_test_dataset(model=None, batch_size=BATCH_SIZE):
    if model is None:
        model = ConvNext(num_input_image_channels=3, num_classes=2).to(DEVICE)
        model.load_state_dict(torch.load("./best_convnext.pth"))
        batch_size = 32
    # After your training loop, add test inference
    print("===== Running inference on test set =====")

    # Load test dataloader
    test_loader = get_dataloader(is_train=False, batch_size=batch_size)
    model.eval()

    all_preds, all_labels = [], []

    with torch.inference_mode():
        for images, labels in tqdm(test_loader, desc="Test Set Inference"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    acc = np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1-score: {f1:.3f}")

    # Save confusion matrix
    os.makedirs("plots", exist_ok=True)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["AD", "NC"])
    disp.plot(cmap="Blues")
    plt.title("Test Set Confusion Matrix")
    plt.savefig("plots/test_confusion_matrix.png")
    plt.close()
    print("Saved test confusion matrix to plots/test_confusion_matrix.png")

    return acc

def main():
    print(f"Using device: {DEVICE}")

    # Load data
    train_loader, val_loader = get_dataloader(is_train=True, batch_size=BATCH_SIZE)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model, loss, optimizer
    model = ConvNext(num_input_image_channels=3, num_classes=2).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # Track metrics
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_val_acc = 0.0
    best_cm = None
    patience = 50  # Stop if no improvement after patience number of epochs
    epochs_no_improve = 0  # Counter for patience

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        val_loss, val_acc, precision, recall, f1, cm = validate(model, val_loader, criterion, epoch)

        # scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"> Epoch {epoch} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | "
            f"P: {precision:.3f}, R: {recall:.3f}, F1: {f1:.3f}")

        # Check for improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_cm = cm
            torch.save(model.state_dict(), "best_convnext.pth")
            print(f"===== 💖 Saved new best model (Val Acc: {best_val_acc:.4f}) =====")
            epochs_no_improve = 0  # reset patience counter
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve}/{patience} epochs.")

        # Early stopping condition
        if epochs_no_improve >= patience:
            print(f"===== Early stopping triggered after {patience} epochs with no improvement =====")
            break

    # Plot results
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)

    # Save confusion matrix of best model
    disp = ConfusionMatrixDisplay(confusion_matrix=best_cm, display_labels=["AD", "NC"])
    disp.plot(cmap="Blues")
    plt.title(f"Best Confusion Matrix (Val Acc: {best_val_acc:.2f}%)")
    cm_path = os.path.join("plots", "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved confusion matrix to {cm_path}")

    run_inference_on_test_dataset(model=model)

if __name__ == "__main__":
    run_inference_on_test_dataset()
