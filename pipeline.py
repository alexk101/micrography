import numpy as np
from skimage.filters import threshold_local
from skimage import img_as_ubyte
import seaborn as sns
import matplotlib as mpl
from sklearn.cluster import DBSCAN
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from utils import remap_labels
from typing import Tuple
import numba

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


def prune_regions(class_sizes: int, labels: np.ndarray) -> np.ndarray:
    print("Pruning regions")
    pruned_labels = labels.copy()
    pruned_labels[~np.isin(labels, list(class_sizes.keys()))] = 0
    # remap so labels are contiguous
    pruned_labels = remap_labels(pruned_labels)