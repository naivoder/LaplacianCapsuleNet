from dataset import ImageDataset, ImagePyramidDataset
from networks import LaplacianNet, CNNNet
from utils import plot_confusion_matrix, plot_curves, plot_roc_auc

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

def train_and_evaluate(dataset_name, train_data, test_data, device, n_epochs=100, cnn_only=False):
    train_indices, val_indices = train_test_split(range(len(train_data)), test_size=0.2, stratify=train_data.targets)
    train_subset = Subset(train_data, train_indices)
    val_subset = Subset(train_data, val_indices)
    
    DatasetClass, NetworkClass = (ImageDataset, CNNNet) if cnn_only else (ImagePyramidDataset, LaplacianNet)
    
    train_loader = DataLoader(DatasetClass(train_subset), batch_size=128, shuffle=True)
    val_loader = DataLoader(DatasetClass(val_subset), batch_size=128, shuffle=False)
    test_loader = DataLoader(DatasetClass(test_data), batch_size=128, shuffle=False)

    num_classes = len(train_data.classes)
    input_shape = train_data[0][0].shape
    
    model = NetworkClass(input_shape, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, amsgrad=True)
    criterion = nn.CrossEntropyLoss()

    best_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader):
            if cnn_only:
                inputs = inputs.to(device)
            else:
                inputs = [inp.to(device) for inp in inputs]
            labels = labels.to(device)
            labels = nn.functional.one_hot(labels, num_classes=num_classes).float()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        y_true = []
        y_pred = []
        
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                if cnn_only:
                    inputs = inputs.to(device)
                else:
                    inputs = [inp.to(device) for inp in inputs]
                labels = labels.to(device)
                labels = nn.functional.one_hot(labels, num_classes=num_classes).float()
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                pred = outputs.argmax(dim=1)
                labels = labels.argmax(dim=1)
                
                correct += (pred == labels).sum().item()
                total += labels.size(0)


        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        val_accuracy = correct / total
        

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), f'weights/{dataset_name}_best.pth')
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter > 10:  # patience
            print("Early stopping triggered")
            break

        
        print(f'[Epoch {epoch + 1}/{n_epochs}] Validation Loss: {val_loss:.4f}  Validation Accuracy: {val_accuracy:.4f}')

    plot_curves(train_losses, val_losses, dataset_name)

    model.load_state_dict(torch.load(f'weights/{dataset_name}_best.pth'))

    model.eval()
    y_true = []
    y_pred = []

    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            if cnn_only:
                inputs = inputs.to(device)
            else:
                inputs = [inp.to(device) for inp in inputs]
            labels = labels.to(device)
            labels = nn.functional.one_hot(labels, num_classes=num_classes).float()
            
            outputs = model(inputs)
            
            pred = outputs.argmax(dim=1)
            labels = labels.argmax(dim=1)
            
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy())


    test_accuracy = correct / total
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    cm_test = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm_test, dataset_name, 'test')
    
    try: # don't kill program in case of failure...
        plot_roc_auc(y_true, y_pred, dataset_name)
    except:
        pass
    
    print(f"Test Accuracy: {test_accuracy:4f}\n")

    return model