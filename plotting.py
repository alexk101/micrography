import time
import numpy as np
import polars as pl
from pathlib import Path
from PIL import Image
import matplotlib as mpl
import cupy as cp
import datashader as ds
from datashader.transfer_functions import shade, spread
import colorcet
import matplotlib.pyplot as plt
import networkx as nx


def save_reconstructed_image(vae_regions: np.ndarray, class_data: pl.DataFrame, sample_dir: Path, name: str, verbose: bool = False):
    """Save reconstructed image with optional benchmarking.
    
    Args:
        vae_regions: Input array of region labels
        class_data: DataFrame containing class predictions
        sample_dir: Directory to save the image
        name: Output filename
        verbose: Whether to print timing information
    """
    timings = {'start': time.perf_counter()}
    
    # Create lookup table for faster class mapping
    timings['prep_start'] = time.perf_counter()
    max_label = vae_regions.max()
    lookup = np.zeros(max_label + 1, dtype=np.int32)
    for label, pred in zip(class_data['class'], class_data['y_pred']):
        lookup[label] = pred
    timings['prep_end'] = time.perf_counter()
    
    # Vectorized reconstruction using lookup table
    timings['reconstruct_start'] = time.perf_counter()
    reconstructed_image = lookup[vae_regions]
    timings['reconstruct_end'] = time.perf_counter()
    
    # Optimize colormap application
    timings['color_start'] = time.perf_counter()
    unique_labels = np.unique(reconstructed_image)
    n_colors = len(unique_labels)
    
    # Pre-compute colormap for unique labels
    cm = mpl.cm.nipy_spectral(np.linspace(0, 1, n_colors))
    color_lookup = np.zeros((unique_labels.max() + 1, 4), dtype=np.float32)
    for i, label in enumerate(unique_labels):
        color_lookup[label] = cm[i]
    
    # Vectorized color mapping
    colored_new_labels = color_lookup[reconstructed_image]
    colored_new_labels = (colored_new_labels * 255).astype(np.uint8)
    timings['color_end'] = time.perf_counter()
    
    # Save image
    timings['save_start'] = time.perf_counter()
    img = Image.fromarray(colored_new_labels)
    img.save(sample_dir/f"{name}.png")
    timings['end'] = time.perf_counter()
    
    # Calculate timing statistics
    if verbose:
        timings.update({
            'prep_time': timings['prep_end'] - timings['prep_start'],
            'reconstruct_time': timings['reconstruct_end'] - timings['reconstruct_start'],
            'color_time': timings['color_end'] - timings['color_start'],
            'save_time': timings['end'] - timings['save_start'],
            'total_time': timings['end'] - timings['start']
        })
        print(pl.DataFrame({'event': timings.keys(), 'time': timings.values()}, strict=False))


# Optional GPU version
def save_reconstructed_image_gpu(vae_regions: np.ndarray, class_data: pl.DataFrame, 
                               sample_dir: Path, name: str, verbose: bool = False):
    """GPU-accelerated version of save_reconstructed_image."""
    timings = {'start': time.perf_counter()}
    
    try:
        # Transfer to GPU
        timings['transfer_to_start'] = time.perf_counter()
        vae_regions_gpu = cp.asarray(vae_regions)
        
        # Create lookup table
        max_label = int(cp.max(vae_regions_gpu).get())
        lookup = cp.zeros(max_label + 1, dtype=cp.int32)
        for label, pred in zip(class_data['class'], class_data['y_pred']):
            lookup[label] = pred
        timings['transfer_to_end'] = time.perf_counter()
        
        # Reconstruction
        timings['reconstruct_start'] = time.perf_counter()
        reconstructed_image_gpu = lookup[vae_regions_gpu]
        timings['reconstruct_end'] = time.perf_counter()
        
        # Transfer back to CPU for coloring
        timings['transfer_from_start'] = time.perf_counter()
        reconstructed_image = cp.asnumpy(reconstructed_image_gpu)
        timings['transfer_from_end'] = time.perf_counter()
        
        # Color mapping (kept on CPU as it's typically not the bottleneck)
        timings['color_start'] = time.perf_counter()
        unique_labels = np.unique(reconstructed_image)
        n_colors = len(unique_labels)
        
        cm = mpl.cm.nipy_spectral(np.linspace(0, 1, n_colors))
        color_lookup = np.zeros((unique_labels.max() + 1, 4), dtype=np.float32)
        for i, label in enumerate(unique_labels):
            color_lookup[label] = cm[i]
        
        colored_new_labels = color_lookup[reconstructed_image]
        colored_new_labels = (colored_new_labels * 255).astype(np.uint8)
        timings['color_end'] = time.perf_counter()
        
        # Save image
        timings['save_start'] = time.perf_counter()
        img = Image.fromarray(colored_new_labels)
        img.save(sample_dir/f"{name}.png")
        timings['end'] = time.perf_counter()
        
    except Exception as e:
        print(f"GPU processing failed: {e}. Falling back to CPU...")
        save_reconstructed_image(vae_regions, class_data, sample_dir, name, verbose)
    
    if verbose:
        timings.update({
            'transfer_to_time': timings['transfer_to_end'] - timings['transfer_to_start'],
            'reconstruct_time': timings['reconstruct_end'] - timings['reconstruct_start'],
            'transfer_from_time': timings['transfer_from_end'] - timings['transfer_from_start'],
            'color_time': timings['color_end'] - timings['color_start'],
            'save_time': timings['end'] - timings['save_start'],
            'total_time': timings['end'] - timings['start']
        })
        print(pl.DataFrame({'event': timings.keys(), 'time': timings.values()}, strict=False))


# Function to choose best implementation based on data size
def save_reconstructed_image_auto(vae_regions: np.ndarray, class_data: pl.DataFrame, 
                                sample_dir: Path, name: str, verbose: bool = False):
    """Automatically choose between CPU and GPU implementation based on data size."""
    # Threshold for GPU usage (adjust based on benchmarking)
    GPU_THRESHOLD = 1000 * 1000  # 1M pixels
    
    if cp.cuda.is_available():
        print(f"GPU is available, vae_regions.size: {vae_regions.size}")
        if vae_regions.size > GPU_THRESHOLD:
            print(f"vae_regions.size > GPU_THRESHOLD, vae_regions.size: {vae_regions.size}")
            save_reconstructed_image_gpu(vae_regions, class_data, sample_dir, name, verbose)
        else:
            print(f"vae_regions.size <= GPU_THRESHOLD, vae_regions.size: {vae_regions.size}")
            save_reconstructed_image(vae_regions, class_data, sample_dir, name, verbose)
    else:
        print("Using CPU")
        save_reconstructed_image(vae_regions, class_data, sample_dir, name, verbose)


def datashader_plot(df: pl.DataFrame, x: str, y: str, value: str) -> Image.Image:
    cvs = ds.Canvas(plot_width=df[x].max(), plot_height=df[y].max())
    agg = cvs.points(df.to_pandas(), x, y, agg=ds.mean(value))
    img = shade(agg, cmap=colorcet.fire, how='linear')
    img = spread(img, px=5)
    return img.to_pil()


def plot_graph_evolution(feat_target: pl.DataFrame, embeddings: np.ndarray, labels: np.ndarray, G: nx.Graph, output_dir: Path):
    # Create a figure with two subplots side by side
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))

    # Original graph visualization
    pos = {i: G.nodes[i]['centroid'] for i in G.nodes}
    nx.draw(G, pos, node_color=feat_target['y_pred'], node_size=1, ax=axes[0, 0])
    axes[0, 0].set_title('Original Graph Structure')

    axes[0, 1].scatter(feat_target['median_x'], feat_target['median_y'], 
                        c=labels, cmap='viridis', s=1)
    axes[0, 1].set_title('Node Embeddings (t-SNE)')
    axes[0, 1].axis('off')

    # Embeddings
    axes[1, 0].scatter(embeddings[:, 0], embeddings[:, 1], 
                        c=labels, cmap='viridis', s=1)
    axes[1, 0].set_title('Node Embeddings (t-SNE)')
    axes[1, 0].axis('off')
    # plt.colorbar(scatter, ax=axes[1, 0], label='Cluster')

    # Plot the embeddings with the original graph
    axes[1, 1].scatter(embeddings[:, 0], embeddings[:, 1], c=feat_target['y_pred'], s=1)
    axes[1, 1].set_title('Node Embeddings with Labeled Molecules')
    axes[1, 1].axis('off')
    fig.savefig(output_dir/"graph_evolution.png")
    plt.tight_layout()
    plt.show()
    