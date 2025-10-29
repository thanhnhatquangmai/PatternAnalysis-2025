"""
train.py
Author: Thanh Nhat Quang Mai - 49347227
Date: 14th October 2025
Description:
    This script trains a ConvNeXt-based classifier on the ADNI dataset using PyTorch.
    It includes functions for training, validation, plotting metrics, and the main training loop.
"""

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
from sklearn.manifold import TSNE
import umap.umap_ as umap
import seaborn as sns

try:
    import google.colab
    IN_COLAB = True
except:
    IN_COLAB = False

if not IN_COLAB:
    from dataset import get_dataloader
    from modules import ConvNeXt

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 450
PATIENCE = 50  # Stop if no improvement after PATIENCE number of epochs
LEARNING_RATE = 4e-3
BATCH_SIZE = 256
WEIGHT_DECAY = 0.01
LABEL_SMOOTHING = 0.1
SAVE_DIR = ""

def train_one_epoch(model, dataloader, criterion, optimizer, epoch):
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): Training data loader.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimization algorithm.
        epoch (int): Current epoch number.

    Returns:
        tuple[float, float]: Average loss and accuracy for this epoch.
    """
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
    """
    Evaluate the model on the validation dataset.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): Validation data loader.
        criterion (nn.Module): Loss function.
        epoch (int): Current epoch number.

    Returns:
        tuple[float, float, float, float, float, np.ndarray]:
            Average loss, accuracy, precision, recall, F1 score, and confusion matrix.
    """
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

def plot_training_curves(train_losses, val_losses, output_dir="plots"):
    """
    Plot and save training & validation loss and accuracy curves over epochs.

    Args:
        train_losses (list[float]): Training loss per epoch.
        val_losses (list[float]): Validation loss per epoch.
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

    plt.tight_layout()

    save_path = os.path.join(SAVE_DIR, output_dir, "training_curves.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved training curves to {save_path}")

def run_inference_on_test_dataset(model=None, batch_size=BATCH_SIZE):
    """
    Run inference on the test dataset, display metrics, confusion matrix,
    and visualize feature embeddings using UMAP.
    """
    if model is None:
        model = ConvNeXt(num_input_image_channels=3, num_classes=2).to(DEVICE)
        model.load_state_dict(torch.load("best_convnext.pth", map_location=DEVICE))
        batch_size = 128

    print("===== Running inference on test set =====")
    test_loader = get_dataloader(is_train=False, batch_size=batch_size)
    model.eval()

    all_preds, all_labels = [], []
    features = []  # Store feature embeddings before classification

    with torch.inference_mode():
        for images, labels in tqdm(test_loader, desc="Test Set Inference"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # === Forward pass (matching the model's internal architecture) ===
            x = images
            for i in range(model.num_stages):
                x = model.downsampling_layers[i](x)
                x = model.stages[i](x)

            # Global average pooling (before final classification)
            x = model.avgpool(x)
            x = x.view(x.size(0), -1)  # flatten (B, feature_dim)
            feats = x.detach().cpu().numpy()  # Store features for UMAP

            outputs = model.classifier(x)  # classification logits
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            features.append(feats)

    # Concatenate features from all batches
    features = np.concatenate(features, axis=0)

    # --- Compute classification metrics ---
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    acc = np.mean(np.array(all_preds) == np.array(all_labels))

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1-score: {f1:.3f}")

    # --- Save confusion matrix ---
    os.makedirs("plots", exist_ok=True)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["AD", "NC"])
    disp.plot(cmap="Blues")
    plt.title(f"Test Set Confusion Matrix (Acc: {acc:.4f})")
    plt.savefig(os.path.join("plots", "test_confusion_matrix.png"))
    plt.close()
    print("Saved test confusion matrix to plots/test_confusion_matrix.png")

    # --- UMAP visualization ---
    print("Running UMAP dimensionality reduction...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
    embeddings_2d = reducer.fit_transform(features)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=embeddings_2d[:, 0],
        y=embeddings_2d[:, 1],
        hue=["AD" if lbl == 0 else "NC" for lbl in all_labels],
        palette={"AD": "red", "NC": "blue"},
        alpha=0.7,
        s=40
    )
    plt.title("UMAP Projection of Test Set Feature Embeddings")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend(title="Label")
    os.makedirs(os.path.join("plots"), exist_ok=True)
    umap_path = os.path.join("plots", "umap.png")
    plt.savefig(umap_path)
    plt.close()
    print(f"Saved UMAP visualization to {umap_path}")


def main():
    """
    Main training pipeline:
    - Loads dataset
    - Initializes model, optimizer, loss and scheduler
    - Trains and validates over multiple epochs
    - Saves best model checkpoint
    - Plots training curves and confusion matrices
    """
    print(f"Using device: {DEVICE}")

    # Load data
    train_loader, val_loader = get_dataloader(is_train=True, batch_size=BATCH_SIZE)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model, loss, optimizer
    model = ConvNeXt(num_input_image_channels=3, num_classes=2).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=EPOCHS)

    # Track metrics
    train_losses, val_losses = [], []

    best_val_loss = float("inf")
    best_cm = None
    epochs_no_improve = 0  # Counter for PATIENCE

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        val_loss, val_acc, precision, recall, f1, cm = validate(model, val_loader, criterion, epoch)

        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"> Epoch {epoch} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | "
            f"P: {precision:.3f}, R: {recall:.3f}, F1: {f1:.3f}")

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_cm = cm
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_convnext.pth"))
            print(f"===== 💖 Saved new best model (Val Acc: {val_acc:.4f}) =====")
            epochs_no_improve = 0  # reset PATIENCE counter
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve}/{PATIENCE} epochs.")

        # Early stopping condition
        if epochs_no_improve >= PATIENCE:
            print(f"===== Early stopping triggered after {PATIENCE} epochs with no improvement =====")
            break

    # Plot results
    plot_training_curves(train_losses, val_losses)

    # Save confusion matrix of best model
    disp = ConfusionMatrixDisplay(confusion_matrix=best_cm, display_labels=["AD", "NC"])
    disp.plot(cmap="Blues")
    plt.title(f"Best Confusion Matrix (Acc: {val_acc:.4f})")
    cm_path = os.path.join(SAVE_DIR, "plots", "val_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved confusion matrix to {cm_path}")

    run_inference_on_test_dataset(model=model)

if __name__ == "__main__":
    run_inference_on_test_dataset()