from dataset import ImageDataset
from networks import LaplacianNet, margin_loss
from utils import plot_confusion_matrix, plot_curves, plot_roc_auc

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import numpy as np

def train_and_evaluate(dataset_name, train_data, test_data, device):
    train_loader = DataLoader(ImageDataset(train_data), batch_size=32, shuffle=True)
    test_loader = DataLoader(ImageDataset(test_data), batch_size=32, shuffle=False)

    num_classes = len(train_data.classes)
    input_shape = train_data[0][0].shape
    model = LaplacianNet(input_shape, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, amsgrad=True)
    criterion = margin_loss

    best_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(100):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = [inp.to(device) for inp in inputs]
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs, labels)
            loss = criterion(labels, outputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        y_true = []
        y_pred = []
        y_score = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = [inp.to(device) for inp in inputs]
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(labels, outputs)
                val_loss += loss.item()

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(outputs.argmax(dim=1).cpu().numpy())
                y_score.extend(outputs.cpu().numpy())

        val_loss /= len(test_loader)
        val_losses.append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), f'weights/{dataset_name}_best.pth')
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter > 10:  # patience
            print("Early stopping triggered")
            break

        print(f'Epoch {epoch + 1}/{100}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

    plot_curves(train_losses, val_losses, dataset_name)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_score = np.array(y_score)

    cm_train = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm_train, dataset_name, 'train')

    plot_roc_auc(y_true, y_score, dataset_name)

    return model
