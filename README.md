# Micrography

Micrography is a python package with the goal of incorporating graph data for the analysis of electron microscope images. 

## Motivation

We want to be able to actively control an electron microscope to analyze regions of a material in the most efficient way possible. In particular, we want to direct the microscope to focus on regions which are "interesting". We can call these areas outlier or defects. 

Using open source STEM images, we can extract molecule locations. We label each pixel defining each region. We then find the centroid of each region, as well as some other features like stdv in the x and y dimensions. We can construct a graph from this by finding the 4 nearest neighbors to each molecule and connecting them with edge, though, this is a naive assumption for now, as not all molecules may have 4 neighbors due to defects. However, we operate on this assumption for now. Using this information from nearest neighbors and the image data, we can learn more meaningful relationships about materials.

## Initial Goals
- [X] Efficiently extract graphs from electron microscope images
- [X] Compare interpretability to traditional Autoencoder
- [X] Explore material graph structure features
- [X] Use GNNs to predict defects in materials
- [ ] Use GNNs to guide an electron microscope to areas of interest

## Current State

We were able to complete quite a lot of work in a short amount of time, but were not able to fully reach our goal of using a GNN to predict unseen regions to guide a microscope to "interesting" regions. We hope to further develop this code to do so in the future, as well as explore more deeply how graphs can help us learn about and guide material imaging and discovery.

## Installation

To run this code, we recommend using a virtual environment. We support conda, uv, and pip. It is only necessary to use one of these methods, so choose the one that you are most comfortable with.

### Conda

We recommend using conda with the mamba solver, as it is much faster than the default conda solver. You can find a distribution through miniforge [repo](https://github.com/conda-forge/miniforge). Once this is installed, you can create an environment with the following command:

**Nvidia GPU**
```bash
mamba env create -f cuda_env.yml
mamba activate micrography
```

**CPU**
```bash
mamba env create -f cpu_env.yml
mamba activate micrography
```

### UV

`uv` is a replacement for `pip` which is faster and more reproducible. You can install it from [here](https://github.com/astral-sh/uv). Once you have `uv` installed, you can create a virtual environment with the following command:

```bash
uv venv
```

### Pip

If you prefer to use `pip`, you can install all dependencies into your current environment with the following command:

```bash
pip install -r requirements.txt
```

However, we recommend using a virtual environment to avoid conflicts with other packages.
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
## Data

Sample data for this repo is taken from [STEM images and associated parameters for Sm-doped BFO](https://doi.org/10.5281/zenodo.4555978).

## Usage

All examples in micrography are written in the form of `marimo` notebooks. These are reproducible and easily version controlled, unlike standard jupyter notebooks. They can additionally be run as standalone python scripts, making them much more ergonomic for development. For more information on `marimo`, see the [documentation](https://marimo.io/).

**NOTE**
This is tentative and will be updated as the project progresses. We are doing most of the current work in the `project.py` file, as well as standardized functions being written in the `pipeline.py` and `utils.py` files. We will move the final work to marimo notebooks for reproducibility and ease of use later.

