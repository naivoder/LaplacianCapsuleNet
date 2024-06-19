import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

def plot_curves(train_losses, val_losses, dataset_name):
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'Training and Validation Loss - {dataset_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'results/{dataset_name}_loss_curves.png')
    plt.close()

def plot_confusion_matrix(cm, dataset_name, split):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {dataset_name} ({split})')
    plt.colorbar()
    plt.savefig(f'results/{dataset_name}_confusion_matrix_{split}.png')
    plt.close()

def plot_roc_auc(y_true, y_score, dataset_name):
    roc_auc = roc_auc_score(y_true, y_score, multi_class='ovr')
    plt.figure()
    plt.plot(roc_auc, label='ROC AUC')
    plt.title(f'ROC AUC Curve - {dataset_name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig(f'results/{dataset_name}_roc_auc.png')
    plt.close()