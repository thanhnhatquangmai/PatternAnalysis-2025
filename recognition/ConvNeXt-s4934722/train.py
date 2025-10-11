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
    ConfusionMatrixDisplay,
    classification_report
)
import numpy as np
import os

from dataset import get_dataloader
from modules import ConvNext

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 30
LEARNING_RATE = 4e-3
BATCH_SIZE = 512

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


