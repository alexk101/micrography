import numpy as np
from skimage.filters import threshold_local
from skimage import img_as_ubyte
import seaborn as sns
import matplotlib as mpl
from sklearn.cluster import DBSCAN
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from typing import Tuple
import numba
import polars as pl
from scipy.interpolate import Rbf

def plot_and_save(data: np.ndarray, target: Path, name: str, ax: plt.Axes, color: bool = False) -> None:
    """Plots the data and saves the full resolution image to the target directory.

    Args:
        data (np.ndarray): Image array
        target (Path): Where to save the image
        name (str): Plot title
        ax (plt.Axes): The axis to plot on
    """
    print(f"Plotting and saving {name}")
    # Plotting
    ax.imshow(data, cmap='nipy_spectral')
    ax.set_axis_off()
    ax.set_title(name)

    img_data = data
    if color:
        cm = mpl.cm.nipy_spectral(np.linspace(0, 1, len(np.unique(data))))
        colored_labels = np.array([cm[x] for x in data])
        img_data = (colored_labels * 255).astype(np.uint8)  # Convert to uint8

    img = Image.fromarray(img_as_ubyte(img_data))
    img.save(target)


def binarize(data: np.ndarray, block_size: int = 31) -> np.ndarray:
    """Binarize the input data using local thresholding.

    Args:
        data (np.ndarray): Input data.
        block_size (int, optional): Block size for local thresholding. Defaults to 31.

    Returns:
        np.ndarray: Binarized data.
    """
    print("Binarizing data")
    thresh = threshold_local(data, block_size=block_size, method='gaussian', offset=0)
    binary = data > thresh
    return binary


def find_atoms(data: np.ndarray) -> np.ndarray:
    """Find atoms in the binary image.

    Args:
        data (np.ndarray): Binary image.

    Returns:
        np.ndarray: Labeled image.
    """

    print("Finding atoms")
    # Use DBSCAN to label each atom
    coords = np.column_stack(np.nonzero(data))  # Extract coordinates of non-zero pixels
    dbscan = DBSCAN(eps=3, min_samples=5)  # Adjust `eps` and `min_samples` as needed
    label_db = dbscan.fit_predict(coords)  # Cluster the coordinates
    
    # Map DBSCAN labels back to the 2D image
    labels = np.zeros_like(data, dtype=int)  # Initialize the 2D label image
    for coord, label in zip(coords, label_db):
        labels[tuple(coord)] = label + 1  # Avoid 0 for background
    return labels


@numba.njit(fastmath=True, parallel=True)
def find_sizes(labels: np.ndarray, cutoff: int= 200) -> Tuple[np.ndarray, dict, int]:
    """Find the size of each region in the labeled image.

    Args:
        data (np.ndarray): Binary image.
        labels (np.ndarray): Labeled image.

    Returns:
        Tuple[np.ndarray, dict, int]: Data about region sizes.
    """
    sizes = np.bincount(labels.ravel())
    class_sizes = dict(zip(np.unique(labels), sizes))
    class_sizes = {k: v for k, v in class_sizes.items() if v < cutoff}
    num_greater = len(sizes[sizes >= cutoff])
    return sizes, class_sizes, num_greater


@numba.njit(fastmath=True, parallel=True)
def prune_regions(class_sizes: dict, labels: np.ndarray) -> np.ndarray:
    """Prune regions based on class sizes and remap labels to be contiguous.

    Args:
        class_sizes (dict): Dictionary of class sizes.
        labels (np.ndarray): Labeled image.

    Returns:
        np.ndarray: Pruned and remapped labeled image.
    """
    print("Pruning regions")
    pruned_labels = labels.copy()
    keys = np.array(list(class_sizes.keys()))

    for i in numba.prange(pruned_labels.shape[0]):
        for j in range(pruned_labels.shape[1]):
            if pruned_labels[i, j] not in keys:
                pruned_labels[i, j] = 0

    # Remap labels to be contiguous
    unique_labels = np.unique(pruned_labels)
    new_labels = np.arange(len(unique_labels))
    label_map = {old: new for old, new in zip(unique_labels, new_labels)}

    for i in numba.prange(pruned_labels.shape[0]):
        for j in range(pruned_labels.shape[1]):
            pruned_labels[i, j] = label_map[pruned_labels[i, j]]

    return pruned_labels


def numba_class_stats(data):
    @numba.njit(fastmath=True, parallel=True)
    def get_stats(data):
        all_outputs = np.zeros((len(np.unique(data)) - 1, 6))
        classes = np.unique(data)[1:]
        for ind in numba.prange(classes.size):
            c = classes[ind]
            x, y = np.where(c == data)
            all_outputs[ind] = [c, x.size, np.median(x), np.median(y), np.std(x), np.std(y)]
        return all_outputs

    output_np = get_stats(data)
    cols_dtypes = {
        'class': pl.Int32, 
        'size': pl.Int32, 
        'median_x': pl.Int32, 
        'median_y': pl.Int32, 
        'std_x': pl.Float32, 
        'std_y': pl.Float32
    }
    output = pl.DataFrame(output_np, schema=cols_dtypes)    
    return output


@numba.njit(fastmath=True, parallel=True)
def reassign_classes(org_img: np.ndarray, org_classes: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Reassign newly learned classes to the original image.

    Args:
        org_img (np.ndarray): A 2D numpy array representing the original image.
        org_classes (np.ndarray): A numpy array representing the original classes.
        y_pred (np.ndarray): A numpy array representing the new classes.

    Returns:
        np.ndarray: A 2D numpy array of the same shape as `org_img` with the new classes.
    """
    class_map = dict(zip(org_classes,y_pred))
    class_map[0] = 0
    output = np.zeros_like(org_img)

    n_row, n_col = org_img.shape
    for row_i in numba.prange(n_row):
        for col_i in numba.prange(n_col):
            output[row_i][col_i] = class_map[org_img[row_i][col_i]]
    return output

### GRAPH FUNCTIONS

@numba.njit(fastmath=True, parallel=True)
def get_nearest_neighbors(centroids: np.ndarray):
    dists = np.zeros((centroids.shape[0], 4))
    indices = np.zeros((centroids.shape[0], 4))
    for i in numba.prange(centroids.shape[0]):
        dist = np.sqrt(np.sum((centroids - centroids[i]) ** 2, axis=1))
        sorted_indices = np.argsort(dist)[1:5]  # Skip the first one because it's the point itself
        dists[i] = dist[sorted_indices]
        indices[i] = sorted_indices
    return dists, indices.astype(np.uint32)


def interpolate_and_plot(df, x_col, y_col, value_col, ax, grid_size=4000, resolution=500, rbf_function='multiquadric'):
    """
    Interpolates data points in a 2D space using Radial Basis Function (RBF) and plots the interpolated image.

    Parameters:
        df (pd.DataFrame): DataFrame containing the input data.
        x_col (str): Name of the column for x-coordinates.
        y_col (str): Name of the column for y-coordinates.
        value_col (str): Name of the column for the continuous values to interpolate.
        grid_size (int): Maximum size of the space (assumes square grid). Default is 4000.
        resolution (int): Number of grid points per axis (lower for faster computation). Default is 500.
        rbf_function (str): RBF function to use. Options: 'multiquadric', 'linear', 'cubic', etc.

    Returns:
        np.ndarray: Interpolated grid of values.
    """
    # Extract columns from DataFrame
    x = df[x_col].values
    y = df[y_col].values
    values = df[value_col].values

    # Create a grid for interpolation
    grid_x, grid_y = np.meshgrid(
        np.linspace(0, grid_size, resolution), 
        np.linspace(0, grid_size, resolution)
    )

    # RBF Interpolation
    rbf = Rbf(x, y, values, function=rbf_function)
    interpolated_values = rbf(grid_x, grid_y)

    # Plot the interpolated image
    ax.imshow(
        interpolated_values, 
        extent=(0, grid_size, 0, grid_size), 
        origin='lower', 
        cmap='viridis', 
        alpha=0.85
    )
    ax.scatter(x, y, c=values, cmap='viridis', s=20, edgecolor='k', label='Input Points')
    ax.title(f"RBF Interpolation: {value_col}")
    ax.set_axis_off()