# Define the Autoencoder model
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

class Autoencoder(nn.Module):
    def __init__(self, num_classes, latent_dim=64):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32, 256),  # Increased layer size for better feature extraction
            nn.GELU(),
            nn.Linear(256, latent_dim),  # Bottleneck layer
            nn.LayerNorm(latent_dim)  # Normalize bottleneck features
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.GELU(),
            nn.Linear(256, 32 * 32 * num_classes),  # Output size for multiclass
            nn.Unflatten(1, (num_classes, 32, 32))  # Unflatten to match target shape
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Training function
def train(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0
    for data in dataloader:
        inputs, targets = data
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast(str(device)):  # Mixed precision training
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
    return total_loss / len(dataloader)

# Accuracy function
def compute_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)  # Multiclass prediction
            correct += (preds == targets).sum().item()
            total += targets.numel()
    return correct / total


def visualize_latent_dist(model: nn.Module, device, dataloader, reduction='pca', num_points=1000) -> plt.Figure:
    """
    Visualize the latent space of an Autoencoder.
    
    Args:
        model (nn.Module): Trained Autoencoder model.
        dataloader (DataLoader): DataLoader containing input data and labels.
        reduction (str): Dimensionality reduction method ('pca' or 'tsne').
        num_points (int): Number of points to sample for visualization.
        
    Returns:
        plt.Figure: A scatter plot of the latent space.
    """
    model.eval()
    latent_vectors = []
    labels = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            latent = model.encoder(inputs)  # Extract latent representations
            latent_vectors.append(latent.cpu())
            labels.append(targets.cpu())
            
    # Combine all batches into a single tensor
    latent_vectors = torch.cat(latent_vectors, dim=0)
    labels = torch.cat(labels, dim=0)
    
    # Subsample if needed
    if latent_vectors.shape[0] > num_points:
        idx = torch.randperm(latent_vectors.shape[0])[:num_points]
        latent_vectors = latent_vectors[idx]
        labels = labels[idx]

    # Convert to numpy for visualization
    latent_vectors = latent_vectors.numpy()
    labels = labels.numpy()
    
    # Apply dimensionality reduction
    if reduction == 'pca':
        reducer = PCA(n_components=2)
    elif reduction == 'tsne':
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
    else:
        raise ValueError("Invalid reduction method. Use 'pca' or 'tsne'.")
    
    reduced_latent = reducer.fit_transform(latent_vectors)

    # Plot the latent space
    fig, ax = plt.subplots(1,1, figsize=(12, 8))
    scatter = ax.scatter(
        reduced_latent[:, 0], reduced_latent[:, 1], c=labels, cmap='tab10', alpha=0.7
    )
    fig.colorbar(scatter, label='Labels')
    ax.set_xlabel('Latent Dimension 1')
    ax.set_ylabel('Latent Dimension 2')
    fig.suptitle(f'Latent Space Visualization ({reduction.upper()})')
    return fig


def visualize_latent_space(model: nn.Module, device, latent_dim, grid_size=10, latent_range=(-3, 3), image_size=(32, 32)) -> plt.Figure:
    """
    Visualize the decoded images from a grid of points in the latent space.
    
    Args:
        model (nn.Module): Trained Autoencoder model.
        grid_size (int): Number of points along each dimension of the grid.
        latent_range (tuple): Range of values for the latent space grid.
        image_size (tuple): Size of the decoded images (H, W).
        
    Returns:
        plt.Figure. Displays a grid of decoded images.
    """
    model.eval()
    
    # Create a grid of latent points
    latent_points = np.linspace(latent_range[0], latent_range[1], grid_size)
    grid_x, grid_y = np.meshgrid(latent_points, latent_points)
    latent_grid = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    
    # Ensure the latent grid has the correct shape
    if latent_grid.shape[1] != latent_dim:
        latent_grid = np.pad(latent_grid, ((0, 0), (0, latent_dim - latent_grid.shape[1])), 'constant')
    
    latent_grid = torch.tensor(latent_grid, dtype=torch.float32).to(device)
    
    # Decode latent points to generate images
    with torch.no_grad():
        decoded_images = model.decoder(latent_grid)
    
    # Move decoded images to CPU and reshape
    decoded_images = decoded_images.cpu().numpy()
    decoded_images = decoded_images.reshape(-1, *image_size)  # Assuming single-channel
    
    # Create a grid to plot the images
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    for i in range(grid_size):
        for j in range(grid_size):
            img_idx = i * grid_size + j
            axes[i, j].imshow(decoded_images[img_idx], cmap='gray')
            axes[i, j].axis('off')
    
    fig.suptitle("Decoded Images from Latent Space Grid", fontsize=16)
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig