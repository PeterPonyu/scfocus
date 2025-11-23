# scFocus  

## **Abstract**

Single-cell transcriptomics captures cell differentiation trajectories through changes in gene expression intensity. However, it is challenging to obtain precise information on the composition of gene sets corresponding to each lineage branch in complex biological systems. The combination of branch probabilities and unsupervised clustering can effectively characterize changes in gene expression intensity, reflecting continuous cell states without relying on prior information. In this study, we propose a analytic algorithm named single-cell (sc)-Focus that divides cell subpopulations based on reinforcement learning and unsupervised branching in low-dimensional latent space of single cells. The lineage component strength of scFocus coincides with the expression regions of hallmark genes, capturing differentiation processes more effectively in comparison to the original low-dimensional latent space and showing a stronger subpopulation discriminative power. Furthermore, scFocus is applied to ten single-cell datasets, including small-scale datasets, common-scale datasets, and multi-batch datasets. This demonstrates its applicability on different types of datasets and showcases its potential in discovering biological changes due to experimental treatments through multi-batch dataset processing. Finally, an online analysis tool based on scFocus was developed, helping researchers and clinicians in the process and visualization of single-cell RNA sequencing data as well as the interpretation of these data through branch probabilities in a streamlined and intuitive way.

## **Graphical Abstract**
<p align="center">  
  <img src="source/_static/Pattern.png" alt="Pattern Image" width="600"/>  
</p>

## **Installation**

[![PyPI](https://img.shields.io/pypi/v/scfocus.svg?color=brightgreen&style=flat)](https://pypi.org/project/scfocus/)

### Requirements

- Python >= 3.9
- Required packages: `scanpy>=1.10.4`, `torch>=1.13.1`, `joblib>=1.2.0`, `tqdm>=4.64.1`, `streamlit>=1.24.0`

### Install from PyPI

``` bash
pip install scfocus
```

### Install from source

```bash
git clone https://github.com/PeterPonyu/scfocus.git
cd scfocus
pip install -e .
```

## **Quick Start**

Here's a simple example to get started with scFocus:

```python
import scanpy as sc
import scfocus

# Load your single-cell data
adata = sc.read_h5ad('your_data.h5ad')

# Preprocess: normalize, log-transform, and compute PCA
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
adata = adata[:, adata.var.highly_variable]
sc.pp.pca(adata)

# Compute UMAP embedding
sc.pp.neighbors(adata, n_neighbors=15)
sc.tl.umap(adata)

# Run scFocus analysis
embedding = adata.obsm['X_umap']
focus = scfocus.focus(embedding, n=6, pct_samples=0.01)
focus.meta_focusing(n=3)
focus.merge_fp2()

# Add focus probabilities to your AnnData object
adata.obsm['focus_probs'] = focus.mfp[0]
for i in range(focus.mfp[0].shape[1]):
    adata.obs[f'Fate_{i}'] = focus.mfp[0][:, i]

# Visualize results
sc.pl.umap(adata, color=[f'Fate_{i}' for i in range(focus.mfp[0].shape[1])])
```

## **Key Parameters**

The `focus` class accepts the following key parameters:

- **f** (array-like): Latent space of the original data (e.g., UMAP or t-SNE coordinates)
- **n** (int, default=8): Number of parallel agents/branches to identify
- **pct_samples** (float, default=0.125): Percentage of samples used in each training step
- **max_steps** (int, default=5): Maximum steps per training episode
- **num_episodes** (int, default=1000): Number of training episodes
- **hidden_dim** (int, default=128): Hidden layer dimension for neural networks
- **res** (float, default=0.05): Resolution for merging similar focus patterns

For a complete list of parameters and their descriptions, see the [API documentation](https://scfocus.readthedocs.io/en/latest/).

## **Documentation**

[![Documentation Status](https://readthedocs.org/projects/scfocus/badge/?version=latest)](https://scfocus.readthedocs.io/en/latest/?badge=latest)

Comprehensive tutorials and API documentation can be found in our [documentation](https://scfocus.readthedocs.io/en/latest/), including:

- Detailed notebooks for different datasets
- Step-by-step tutorials
- API reference

## **Streamlit Web Interface**

scFocus provides an interactive web interface for easy data analysis without coding.

### Online Access

Access the hosted version at [scfocus.streamlit.app](https://scfocus.streamlit.app/).

### Local Interface

Launch the local web interface (Linux and macOS):

```bash
scfocus ui
```

### Using the Web Interface

1. **Upload Data**: Support for `.h5ad` files or 10x Genomics format (matrix.mtx, features.tsv, barcodes.tsv)
2. **Configure Parameters**:
   - Number of highly variable genes (200-5000, default: 2000)
   - Number of neighbors for UMAP (2-50, default: 15)
   - Minimum distance for UMAP (0.0-2.0, default: 0.5)
   - Number of branches (2-10, default: 6)
3. **Process**: Click "Process" to run the full analysis pipeline
4. **Visualize**: View UMAP plots colored by cell fate probabilities
5. **Download**: Export processed data as `.h5ad` file

Example datasets are available in the `data/` folder of the repository.

## **Command Line Interface (CLI)**

### Available Commands

```bash
# Launch web interface
scfocus ui

# Process single-cell data (coming soon)
# scfocus process --input data.h5ad --output results.h5ad

# Visualize results (coming soon)
# scfocus visualize --input results.h5ad
```

Note: `process` and `visualize` commands are planned for future releases.

## **Workflow Overview**

The typical scFocus workflow consists of:

1. **Preprocessing**: Normalize and log-transform the data, select highly variable genes
2. **Dimensionality Reduction**: Compute PCA and UMAP/t-SNE embeddings
3. **scFocus Analysis**: Apply reinforcement learning to identify lineage branches
4. **Merge Patterns**: Consolidate similar focus patterns
5. **Visualization**: Display cell fate probabilities and branch assignments

## **Troubleshooting**

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'torch'`
- **Solution**: Ensure PyTorch is installed: `pip install torch>=1.13.1`

**Issue**: CUDA out of memory error
- **Solution**: The algorithm automatically uses CPU if GPU is unavailable. For large datasets, consider reducing `n` (number of agents) or `pct_samples`.

**Issue**: Streamlit command not found
- **Solution**: Install streamlit: `pip install streamlit>=1.24.0`

**Issue**: Analysis is very slow
- **Solution**: 
  - Reduce `num_episodes` (default 1000) for faster but less refined results
  - Decrease `n` (number of agents) to reduce computational load
  - Use GPU if available for faster training

### Getting Help

- Check the [documentation](https://scfocus.readthedocs.io/en/latest/)
- Open an issue on [GitHub](https://github.com/PeterPonyu/scfocus/issues)
- Review example notebooks in the documentation

## **Development**

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/PeterPonyu/scfocus.git
cd scfocus

# Install in development mode
pip install -e .

# Install additional development dependencies (if any)
pip install -r requirements.txt
```

### Building Documentation

```bash
cd source
make html
```

The documentation will be built in `build/html/`.

## **License**
<p>
    <a href="https://choosealicense.com/licenses/mit/" target="_blank">
        <img alt="license" src="https://img.shields.io/github/license/PeterPonyu/scfocus?style=flat-square&color=brightgreen"/>
    </a>
</p>


## **Cite**

- Chen, C., Fu, Z., Yang, J., Chen, H., Huang, J., Qin, S., Wang, C., & Hu, X. (2025). **scFocus: Detecting Branching Probabilities in Single-cell Data with SAC**. *Computational and Structural Biotechnology Journal*. [https://doi.org/10.1016/j.csbj.2025.04.036](https://doi.org/10.1016/j.csbj.2025.04.036)

