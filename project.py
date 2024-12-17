import marimo

__generated_with = "0.10.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""Download Data""")
    return


@app.cell
def _():
    import utils

    utils.download_data()
    compositions, uc_params, imgdata = utils.parse_files()
    SBFO_data = utils.process_data(compositions, uc_params, imgdata)
    return SBFO_data, compositions, imgdata, uc_params, utils


@app.cell
def _(SBFO_data):
    import matplotlib.pyplot as plt
    from PIL import Image
    from pathlib import Path
    target = Path('sample_imgs')
    target.mkdir(exist_ok=True)
    (fig, axes) = plt.subplots(2, len(SBFO_data), figsize=(18, 9))
    for (i, (im1, ax, ax_2)) in enumerate(zip(SBFO_data, axes[0], axes[1])):
        ax.imshow(im1['image'])
        ax.set_axis_off()
        ax_2.hist(im1['image'][:].ravel(), bins=256, range=(0, 255), fc='k', ec='k')
        if not (target / f'sample_{i}.png').exists():
            img = Image.fromarray(im1['image'][:])
            img = img.convert('L')
            img.save(target / f'sample_{i}.png')
    fig
    return Image, Path, ax, ax_2, axes, fig, i, im1, img, plt, target


@app.cell
def _(Image, SBFO_data, plt, target):
    from skimage.morphology import binary_erosion, binary_dilation, disk
    import numpy as np
    from skimage.filters import threshold_local
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max
    from skimage import img_as_ubyte
    import seaborn as sns
    from scipy import ndimage as ndi
    import matplotlib as mpl

    images = {}
    (fig_1, axes_1) = plt.subplots(6, len(SBFO_data[:2]), figsize=(18, 18))
    for (j, (im2, (ax1, ax2, ax3, ax4, ax5, ax6))) in enumerate(zip(SBFO_data[:2], axes_1.T)):
        # Binary Threshold
        data = {}
        thresh = threshold_local(im2['image'][:], block_size=31, method='gaussian', offset=0)
        binary = im2['image'] > thresh
        data['binary'] = binary

        # Plotting
        ax1.imshow(binary, cmap='gray')
        ax1.set_axis_off()
        ax1.set_title('Image mask')
        img_1 = Image.fromarray(img_as_ubyte(binary))
        img_1.save(target / f'sample_{j}_binary.png')

        # Binary erosion
        eroded = binary_erosion(binary, disk(2))
        data['eroded'] = eroded

        # Plotting
        ax2.imshow(eroded, cmap='gray')
        ax2.set_axis_off()
        ax2.set_title('Eroded mask')
        img_1 = Image.fromarray(img_as_ubyte(eroded))
        img_1.save(target / f'sample_{j}_eroded.png')

        # Binary dilation
        dilated = binary_dilation(eroded, disk(1))
        data['dilated'] = dilated

        # Plotting
        img_1 = Image.fromarray(img_as_ubyte(dilated))
        img_1.save(target / f'sample_{j}_dilated.png')
        ax3.imshow(dilated, cmap='gray')
        ax3.set_axis_off()
        ax3.set_title('Dilated mask')

        # Watershed
        distance = ndi.distance_transform_edt(dilated)
        coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=eroded)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        (markers, _) = ndi.label(mask)
        labels = watershed(-distance, markers, mask=distance)
        data['clf'] = labels

        # Plotting
        ax4.imshow(labels, cmap='nipy_spectral')
        ax4.set_axis_off()
        ax4.set_title(f'Number of regions: {len(np.unique(labels))}')
        cm = mpl.cm.nipy_spectral(np.linspace(0, 1, len(np.unique(labels))))
        colored_labels = np.array([cm[label] for label in labels])
        colored_labels = (colored_labels * 255).astype(np.uint8)
        img_1 = Image.fromarray(colored_labels)
        img_1.save(target / f'sample_{j}_clf.png')

        
        cutoff = 200
        sizes = np.bincount(labels.ravel())
        class_sizes = dict(zip(np.unique(labels), sizes))
        class_sizes = {k: v for (k, v) in class_sizes.items() if v < cutoff}
        num_greater = len(sizes[sizes >= cutoff])
        sizes = np.array(list(class_sizes.values()))
        print(f'Number of regions with size >= {cutoff}: {num_greater}')

        # Plotting
        sns.histplot(sizes, bins=50, ax=ax5)
        ax5.set_title('Region size distribution')
        pruned_labels = labels.copy()
        pruned_labels[~np.isin(labels, list(class_sizes.keys()))] = 0
        data['pruned'] = pruned_labels

        # Plotting
        ax6.imshow(pruned_labels, cmap='nipy_spectral')
        ax6.set_axis_off()
        ax6.set_title(f'Number of regions: {len(np.unique(pruned_labels))}')
        colored_pruned_labels = np.array([cm[label] for label in pruned_labels])
        colored_pruned_labels = (colored_pruned_labels * 255).astype(np.uint8)
        img_1 = Image.fromarray(colored_pruned_labels)
        img_1.save(target / f'sample_{j}_clf_pruned.png')
        images[j] = data
    fig_1
    return (
        ax1,
        ax2,
        ax3,
        ax4,
        ax5,
        ax6,
        axes_1,
        binary,
        binary_dilation,
        binary_erosion,
        class_sizes,
        cm,
        colored_labels,
        colored_pruned_labels,
        coords,
        cutoff,
        data,
        dilated,
        disk,
        distance,
        eroded,
        fig_1,
        im2,
        images,
        img_1,
        img_as_ubyte,
        j,
        labels,
        markers,
        mask,
        mpl,
        ndi,
        np,
        num_greater,
        peak_local_max,
        pruned_labels,
        sizes,
        sns,
        thresh,
        threshold_local,
        watershed,
    )


@app.cell
def _(np):
    import polars as pl
    import numba
    # Use dispersion of the region sizes to determine outliers

    # Numba Optimized
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
            'class': pl.UInt32, 
            'size': pl.UInt32, 
            'median_x': pl.UInt32, 
            'median_y': pl.UInt32, 
            'std_x': pl.Float32, 
            'std_y': pl.Float32
        }
        output = pl.DataFrame(output_np, schema=cols_dtypes)    
        return output
    return numba, numba_class_stats, pl


@app.cell
def _(images, numba_class_stats):
    class_data = numba_class_stats(images[0]['pruned'])
    return (class_data,)


@app.cell
def _(class_data, plt, sns):
    (fig_2, axes_2) = plt.subplots(2, len(class_data.columns) // 2, figsize=(18, 9))
    for col, ax_7 in zip(class_data.columns, axes_2.flatten()):
        sns.histplot(class_data[col], ax=ax_7, kde=True)
        ax_7.set_title(col)
    fig_2
    return ax_7, axes_2, col, fig_2


@app.cell
def _(class_data, np, numba):
    # Calculate nearest neighbor distances of 4 nearest neighbors
    # The data consists of a 2d array where the non-zero values are the regions
    # The distance should be the euclidean distance between the centroids of the regions
    # The output should be a 2d array of shape (n, 4) where n is the number of regions
    # The output should be the distances and the indices of the nearest neighbors

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

    centroids = np.array(
        [
            class_data['median_x'].to_numpy(), 
            class_data['median_y'].to_numpy()
        ]
    ).T
    dists, indices = get_nearest_neighbors(centroids)
    print(dists, indices)
    print(dists.shape, indices.shape)
    return centroids, dists, get_nearest_neighbors, indices


@app.cell
def _(centroids, class_data, dists, indices, plt):
    from networkx import Graph
    import networkx as nx

    def construct_graph(dists, indices, centroids):
        G = Graph()
        for (i, (dist, idx)) in enumerate(zip(dists, indices)):
            for (d, j) in zip(dist, idx):
                G.add_edge(i, j, weight=d)
            G.nodes[i]['centroid'] = centroids[i]
            G.nodes[i]['size'] = class_data['size'][i]
        return G
    G = construct_graph(dists, indices, centroids)
    print(G.nodes[0])
    (fig_3, ax_3) = plt.subplots(figsize=(20, 20))
    pos = {i: G.nodes[i]['centroid'] for i in G.nodes}
    nx.draw(G, pos, node_size=1, ax=ax_3)
    plt.show()
    return G, Graph, ax_3, construct_graph, fig_3, nx, pos


@app.cell
def _(class_data, sns):
    sns.kdeplot(data=class_data, x='std_x', y='std_y', fill=True, cmap='viridis')
    return


@app.cell
def _(np, numba):
    @numba.njit(fastmath=True, parallel=True)
    def reassign_classes(org_img: np.ndarray, org_classes: np.ndarray, y_pred: np.ndarray):
        class_map = dict(zip(org_classes, y_pred))
        class_map[0] = 0
        output = np.zeros_like(org_img)
        (n_row, n_col) = org_img.shape
        for row_i in numba.prange(n_row):
            for col_i in numba.prange(n_col):
                output[row_i][col_i] = class_map[org_img[row_i][col_i]]
        return output
    return (reassign_classes,)


@app.cell
def _(Image, class_data, cm, images, mpl, np, plt, reassign_classes):
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    print(class_data)

    feats = ['std_x', 'std_y', 'size']
    y_pred_2 = GaussianMixture(n_components=2, random_state=42).fit_predict(class_data[feats]) + 1
    org_img_2 = images[0]['pruned'].astype(np.uint32)
    org_classes_2 = class_data['class'].to_numpy()
    new_labels_2 = reassign_classes(org_img_2, org_classes_2, y_pred_2)
    print(np.unique(new_labels_2))
    (fig_4, ax_1) = plt.subplots(1, 1, figsize=(9, 9))
    ax_1.imshow(new_labels_2, cmap='nipy_spectral')
    cm_3 = mpl.cm.nipy_spectral(np.linspace(0, 1, len(np.unique(new_labels_2))))
    colored_new_labels_2 = np.array([cm[label] for label in new_labels_2])
    colored_new_labels_2 = (colored_new_labels_2 * 255).astype(np.uint8)
    img_gmm = Image.fromarray(colored_new_labels_2)
    img_gmm.save(f'sample_{0}_gmm.png')
    return (
        GaussianMixture,
        KMeans,
        ax_1,
        cm_3,
        colored_new_labels_2,
        feats,
        fig_4,
        img_gmm,
        new_labels_2,
        org_classes_2,
        org_img_2,
        y_pred_2,
    )


@app.cell
def _(centroids, construct_graph, dists, indices, nx, plt, y_pred_2):
    G_2 = construct_graph(dists, indices, centroids)
    (fig_5, ax_8) = plt.subplots(figsize=(20, 20))
    pos_2 = {i: G_2.nodes[i]['centroid'] for i in G_2.nodes}
    nx.draw(G_2, pos_2, node_color=y_pred_2, node_size=1, ax=ax_8)
    fig_5
    return G_2, ax_8, fig_5, pos_2


@app.cell
def _(Image, class_data, cm, images, mpl, np, plt, reassign_classes):
    x_var = class_data['std_x'].to_numpy() > 2.3
    y_var = class_data['std_y'].to_numpy() > 2.3
    y_pred_1 = x_var & y_var
    y_pred_1 = y_pred_1.astype(np.uint32) + 1
    org_img = images[0]['pruned'].astype(np.uint32)
    org_classes = class_data['class'].to_numpy()
    new_labels = reassign_classes(org_img, org_classes, y_pred_1)
    print(np.unique(new_labels))

    (fig_6, ax_4) = plt.subplots(1, 1, figsize=(9, 9))
    ax_4.set_axis_off()
    ax_4.imshow(new_labels, cmap='nipy_spectral')
    cm_2 = mpl.cm.nipy_spectral(np.linspace(0, 1, len(np.unique(new_labels))))
    colored_new_labels = np.array([cm[label] for label in new_labels])
    colored_new_labels = (colored_new_labels * 255).astype(np.uint8)
    img_4 = Image.fromarray(colored_new_labels)
    img_4.save(f'sample_{0}_x_std.png')
    fig_6
    return (
        ax_4,
        cm_2,
        colored_new_labels,
        fig_6,
        img_4,
        new_labels,
        org_classes,
        org_img,
        x_var,
        y_pred_1,
        y_var,
    )


if __name__ == "__main__":
    app.run()
