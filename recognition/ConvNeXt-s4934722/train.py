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

from dataset import get_dataloader
from modules import ConvNext

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 300
LEARNING_RATE = 4e-3

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
    accuracy = 100 * correct / total
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
            total_loss += loss.item() * batch_size
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += batch_size

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / loss
    accuracy = 100 * correct / total

    # Compute additional metrics
    precision = precision_score(y_true=all_labels, y_pred=all_preds)
    recall = recall_score(y_true=all_labels, y_pred=all_preds)
    f1 = f1_score(y_true=all_labels, y_pred=all_preds)
    cm = confusion_matrix(y_true=all_labels, y_pred=all_preds)

    return avg_loss, accuracy, precision, recall, f1, cm
