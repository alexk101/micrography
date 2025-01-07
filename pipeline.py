import numpy as np
from skimage.filters import threshold_local
from skimage import img_as_ubyte
import seaborn as sns
import matplotlib as mpl
from sklearn.cluster import DBSCAN
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from typing import Tuple, Dict
import numba
import polars as pl
from scipy.interpolate import Rbf
from networkx import Graph
import cupy as cp
import time

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


def numba_class_stats(data: np.ndarray, device: str = 'cpu', verbose: bool = False) -> pl.DataFrame:
    """Calculate class statistics using either GPU or CPU implementation.
    
    Args:
        data: Input image array
        use_gpu: Whether to attempt GPU processing first
        benchmark: Whether to return timing information
    
    Returns:
        DataFrame with class statistics, and optionally timing information
    """
    def get_stats_gpu(data: np.ndarray) -> np.ndarray:
        start_times = {'gpu_transfer_to': time.perf_counter()}
        
        # Move data to GPU
        data_gpu = cp.asarray(data)
        start_times['gpu_compute'] = time.perf_counter()
        
        # Get unique classes (excluding 0)
        classes = cp.unique(data_gpu)[1:]
        n_classes = len(classes)
        
        # Initialize output array
        all_outputs = cp.zeros((n_classes, 6), dtype=cp.float32)
        
        # Process each class in parallel on GPU
        for i, c in enumerate(classes):
            # Get positions where class c exists
            positions = cp.where(data_gpu == c)
            x_pos = positions[0]
            y_pos = positions[1]
            size = len(x_pos)
            
            # Calculate statistics
            median_x = cp.median(x_pos)
            median_y = cp.median(y_pos)
            std_x = cp.std(x_pos)
            std_y = cp.std(y_pos)
            
            all_outputs[i] = [c, size, median_x, median_y, std_x, std_y]
        
        start_times['gpu_transfer_from'] = time.perf_counter()
        # Move results back to CPU
        result = cp.asnumpy(all_outputs)
        start_times['gpu_end'] = time.perf_counter()
        
        return result, start_times

    def get_stats_cpu(data: np.ndarray) -> np.ndarray:
        start_times = {'cpu_start': time.perf_counter()}
        
        @numba.njit(fastmath=True, parallel=True)
        def compute_stats(data):
            # Pre-compute unique classes excluding 0
            classes = np.unique(data)[1:]
            n_classes = len(classes)
            
            # Create class index mapping for faster lookup
            class_to_idx = np.zeros(classes.max() + 1, dtype=np.int32)
            for i, c in enumerate(classes):
                class_to_idx[c] = i
                
            # First pass: count sizes to pre-allocate arrays
            sizes = np.zeros(n_classes, dtype=np.int32)
            for i in numba.prange(data.shape[0]):
                for j in range(data.shape[1]):
                    val = data[i, j]
                    if val != 0:
                        sizes[class_to_idx[val]] += 1
            
            # Pre-allocate position arrays for each class
            x_positions = [np.zeros(sizes[i], dtype=np.int32) for i in range(n_classes)]
            y_positions = [np.zeros(sizes[i], dtype=np.int32) for i in range(n_classes)]
            counters = np.zeros(n_classes, dtype=np.int32)
            
            # Second pass: collect positions
            for i in numba.prange(data.shape[0]):
                for j in range(data.shape[1]):
                    val = data[i, j]
                    if val == 0:
                        continue
                    idx = class_to_idx[val]
                    pos = counters[idx]
                    x_positions[idx][pos] = i
                    y_positions[idx][pos] = j
                    counters[idx] += 1
            
            # Calculate statistics
            all_outputs = np.zeros((n_classes, 6))
            for i in range(n_classes):
                x_pos = x_positions[i]
                y_pos = y_positions[i]
                
                # Sort for median calculation
                x_pos.sort()
                y_pos.sort()
                
                mid = sizes[i] // 2
                median_x = x_pos[mid] if sizes[i] % 2 == 1 else (x_pos[mid-1] + x_pos[mid]) / 2
                median_y = y_pos[mid] if sizes[i] % 2 == 1 else (y_pos[mid-1] + y_pos[mid]) / 2
                
                # Calculate std
                std_x = np.std(x_pos)
                std_y = np.std(y_pos)
                
                all_outputs[i] = [classes[i], sizes[i], median_x, median_y, std_x, std_y]
                
            return all_outputs

        result = compute_stats(data)
        start_times['cpu_end'] = time.perf_counter()
        return result, start_times

    # Choose processing path and handle results
    timings = {}
    if device == 'gpu' and cp.cuda.is_available():
        try:
            output_np, gpu_times = get_stats_gpu(data)
            timings.update(gpu_times)
            timings['method'] = 'gpu'
        except (cp.cuda.memory.OutOfMemoryError, Exception) as e:
            print(f"GPU processing failed: {e}. Falling back to CPU...")
            output_np, cpu_times = get_stats_cpu(data)
            timings.update(cpu_times)
            timings['method'] = 'cpu_fallback'
    else:
        output_np, cpu_times = get_stats_cpu(data)
        timings.update(cpu_times)
        timings['method'] = 'cpu'

    # Create DataFrame
    cols_dtypes = {
        'class': pl.Int32, 
        'size': pl.Int32, 
        'median_x': pl.Int32,
        'median_y': pl.Int32,
        'std_x': pl.Float32, 
        'std_y': pl.Float32
    }
    output = pl.DataFrame(output_np, schema=cols_dtypes)

    # Calculate timing statistics
    if timings['method'] == 'gpu':
        timings['transfer_to_time'] = timings['gpu_compute'] - timings['gpu_transfer_to']
        timings['compute_time'] = timings['gpu_transfer_from'] - timings['gpu_compute']
        timings['transfer_from_time'] = timings['gpu_end'] - timings['gpu_transfer_from']
        timings['total_time'] = timings['gpu_end'] - timings['gpu_transfer_to']
    else:
        timings['total_time'] = timings['cpu_end'] - timings['cpu_start']

    if verbose:
        print(f"numba_class_stats took {timings['total_time']} seconds")
        print(pl.DataFrame({'event': timings.keys(), 'time': timings.values()}, strict=False))
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
def get_nearest_neighbors(centroids: np.ndarray, k: int = 4):
    dists = np.zeros((centroids.shape[0], k))
    indices = np.zeros((centroids.shape[0], k))
    for i in numba.prange(centroids.shape[0]):
        dist = np.sqrt(np.sum((centroids - centroids[i]) ** 2, axis=1))
        sorted_indices = np.argsort(dist)[1:k+1]  # Skip the first one because it's the point itself
        dists[i] = dist[sorted_indices]
        indices[i] = sorted_indices
    return dists, indices.astype(np.uint32)

def find_neighbors(class_features: pl.DataFrame, k: int = 4):
    centroids = np.array(
        [
            class_features['median_x'].to_numpy(), 
            class_features['median_y'].to_numpy()
        ]
    ).T
    dists, indices = get_nearest_neighbors(centroids, k)

    # Let's also add the 4 nearest neighbors to the class features
    class_features = class_features.with_columns([pl.Series(f'nn_{i+1}_dist', dists[:, i]) for i in range(k)])
    return (dists, indices), class_features

def construct_graph(dists, indices, class_feat) -> Graph:
    G = Graph()
    centroids = class_feat.select(['median_x', 'median_y']).to_numpy()
    for i, (dist, idx) in enumerate(zip(dists, indices)):
        for d, j in zip(dist, idx):
            G.add_edge(i, j, weight=d)
        
        G.nodes[i]['centroid'] = centroids[i]
        G.nodes[i]['size'] = class_feat['size'][i]
        G.nodes[i]['std_x'] = class_feat['std_x'][i]
        G.nodes[i]['std_y'] = class_feat['std_y'][i]
        G.nodes[i]['class'] = class_feat['y_pred'][i]
    return G

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


def transfer_classes(class_data: pl.DataFrame, y_pred: np.ndarray, device: str = 'cpu', verbose: bool = False) -> pl.DataFrame:
    """Transfer predicted classes to molecules with optional benchmarking.

    Args:
        class_data: DataFrame containing molecule data
        y_pred: Predicted classes array
        device: 'cpu' or 'cuda'
        benchmark: Whether to return timing information

    Returns:
        DataFrame with transferred classes, and optionally timing information
    """
    @numba.njit(parallel=True, fastmath=True)
    def _test_assign(molecules_x, molecules_y, y_pred):
        n = len(molecules_x)
        result = np.empty(n, dtype=y_pred.dtype)
        
        for i in numba.prange(n):
            result[i] = y_pred[molecules_x[i], molecules_y[i]]
        
        return result
    
    @numba.njit(parallel=True, fastmath=True)
    def _test_assign_chunked(molecules_x, molecules_y, y_pred):
        n = len(molecules_x)
        result = np.empty(n, dtype=y_pred.dtype)
        chunk_size = 1000
        
        for chunk in numba.prange((n + chunk_size - 1) // chunk_size):
            start = chunk * chunk_size
            end = min(start + chunk_size, n)
            for i in range(start, end):
                result[i] = y_pred[molecules_x[i], molecules_y[i]]
        
        return result
    
    def _test_assign_cuda(molecules_x, molecules_y, y_pred):
        timings = {'gpu_transfer_to': time.perf_counter()}
        
        # Transfer arrays to GPU
        molecules_x_gpu = cp.asarray(molecules_x)
        molecules_y_gpu = cp.asarray(molecules_y) 
        y_pred_gpu = cp.asarray(y_pred)
        result_gpu = cp.empty(len(molecules_x), dtype=y_pred.dtype)
        
        timings['gpu_compute'] = time.perf_counter()
        
        # Create indexing array and perform lookup
        idx = cp.arange(len(molecules_x))
        result_gpu = y_pred_gpu[molecules_x_gpu[idx], molecules_y_gpu[idx]]
        
        timings['gpu_transfer_from'] = time.perf_counter()
        
        # Transfer result back to CPU
        result = cp.asnumpy(result_gpu)
        
        timings['gpu_end'] = time.perf_counter()
        return result, timings

    # Start timing
    timings = {'start': time.perf_counter()}

    # Make sure we have writable copies of our input arrays
    timings['copy_start'] = time.perf_counter()
    molecules = class_data['class'].to_numpy().copy()
    molecules_x = class_data['median_x'].to_numpy().copy()
    molecules_y = class_data['median_y'].to_numpy().copy()
    y_pred = y_pred.copy()
    timings['copy_end'] = time.perf_counter()
    
    if device == 'cuda':
        print("Using CUDA")
        if cp.cuda.is_available():
            try:
                output, gpu_times = _test_assign_cuda(molecules_x, molecules_y, y_pred)
                timings.update(gpu_times)
                timings['method'] = 'gpu'
            except (cp.cuda.memory.OutOfMemoryError, Exception) as e:
                print(f"GPU processing failed: {e}. Falling back to CPU...")
                timings['compute_start'] = time.perf_counter()
                output = _test_assign_chunked(molecules_x, molecules_y, y_pred)
                timings['compute_end'] = time.perf_counter()
                timings['method'] = 'cpu_fallback'
        else:
            print("CUDA not available, falling back to CPU")
            timings['compute_start'] = time.perf_counter()
            output = _test_assign_chunked(molecules_x, molecules_y, y_pred)
            timings['compute_end'] = time.perf_counter()
            timings['method'] = 'cpu_fallback'
    else:
        print("Using CPU")
        timings['compute_start'] = time.perf_counter()
        output = _test_assign_chunked(molecules_x, molecules_y, y_pred)
        timings['compute_end'] = time.perf_counter()
        timings['method'] = 'cpu'

    # DataFrame creation timing
    timings['df_start'] = time.perf_counter()
    result = pl.DataFrame({
        'class': molecules,
        'y_pred': output
    })
    timings['end'] = time.perf_counter()

    # Calculate timing statistics
    if timings['method'] == 'gpu':
        timings['transfer_to_time'] = timings['gpu_compute'] - timings['gpu_transfer_to']
        timings['compute_time'] = timings['gpu_transfer_from'] - timings['gpu_compute']
        timings['transfer_from_time'] = timings['gpu_end'] - timings['gpu_transfer_from']
    else:
        timings['compute_time'] = timings['compute_end'] - timings['compute_start']
    
    timings['copy_time'] = timings['copy_end'] - timings['copy_start']
    timings['df_creation_time'] = timings['end'] - timings['df_start']
    timings['total_time'] = timings['end'] - timings['start']

    if verbose:
        print(f"transfer_classes took {timings['total_time']} seconds")
        print(pl.DataFrame({'event': timings.keys(), 'time': timings.values()}, strict=False))
    return result
