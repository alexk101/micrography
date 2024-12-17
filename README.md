# Micrography

Micrography is a python package with the goal of incorporating graph data for the analysis of electron microscope images. 

## Initial Goals
- [ ] efficiently extract graphs from electron microscope images
- [ ] use GNNs to predict defects in materials
- [ ] use GNNs to guide the electron microscope to areas of interest

## Installation

To run this code, we recommend using a some kind of virtual environment. We support conda, uv, and pip. It is only necessary to use one of these methods, so choose the one that you are most comfortable with.

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
pip install .
```

However, we recommend using a virtual environment to avoid conflicts with other packages.
```bash
python -m venv venv
source venv/bin/activate
pip install .
```
## Data

Sample data for this repo is taken from [STEM images and associated parameters for Sm-doped BFO](https://doi.org/10.5281/zenodo.4555978).

## Usage

All examples in micrography are written in the form of `marimo` notebooks. These are reproducible and easily version controlled, unlike standard jupyter notebooks. They can additionally be run as standalone python scripts, making them much more ergonomic for development. For more information on `marimo`, see the [documentation](https://marimo.io/).