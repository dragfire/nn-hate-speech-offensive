
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchmetrics import ConfusionMatrix
import itertools

def plot_losses(train_losses, val_losses):
    plt.plot(train_losses, label='Train loss')
    plt.plot(val_losses, label='Val loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss over Iteration')
    plt.legend()
    plt.show()

def plot_pca(vocab, embedding_weights):
    class_labels = [i for i in range(len(vocab))] 
    class_names = list(vocab)

    # Convert the embedding weights to a NumPy array
    embedding_weights_np = embedding_weights.numpy()

    # Calculate the mean of the embeddings
    mean_embedding = torch.mean(embedding_weights, dim=0)

    # Center the embeddings
    centered_embeddings = embedding_weights - mean_embedding

    # Compute the covariance matrix
    cov_matrix = torch.mm(centered_embeddings.t(), centered_embeddings)

    # Perform Singular Value Decomposition (SVD) on the covariance matrix
    U, S, V = torch.svd(cov_matrix)

    # Select the top two principal components
    pca_components = U[:, :2]

    # Project the embeddings onto the first two principal components
    embedded_points = torch.mm(centered_embeddings, pca_components)

    # Convert the embeddings and class labels to NumPy arrays
    embedded_points_np = embedded_points.numpy()
    class_labels_np = np.array(class_labels)

    # Plot the embeddings with color-coded classes
    plt.figure(figsize=(20, 15))
    plt.scatter(embedded_points_np[:, 0], embedded_points_np[:, 1], c=class_labels_np, cmap='viridis', s=5)
    for i, label in enumerate(class_labels):
        plt.text(embedded_points_np[i, 0], embedded_points_np[i, 1], class_names[label], fontsize=10, ha='center', va='center')

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA Visualization of Embedding Layer with Color-Coded Classes")
    plt.show()

# class_labels = ['hate speech', 'offensive language', 'neutral']

def plot_cf_matrix(batches, class_labels, model, device='cpu'):
  confusion_matrix = ConfusionMatrix(task='multiclass', num_classes=len(class_labels))
  with torch.no_grad():
    for xb, yb in batches:
        xb = xb.to(device)
        preds = torch.tensor(model(xb).topk(1).indices.reshape(1, -1).cpu().numpy()[0])
        confusion_matrix.update((preds), yb)

  result = confusion_matrix.compute()
  cm = result.numpy()

  ticks = np.arange(0, len(class_labels))
  plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
  plt.colorbar()
  plt.xticks(ticks, class_labels)
  plt.yticks(ticks, class_labels)

  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, str(cm[i,j]))

  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.tight_layout()