scFocus documentation
=====================

scFocus
=======

About scFocus
-------------

**scFocus** is an innovative approach that leverages reinforcement learning algorithms to conduct biologically meaningful analyses. By utilizing branch probabilities, scFocus enhances cell subtype discrimination without requiring prior knowledge of differentiation starting points or cell subtypes.

To identify distinct lineage branches within single-cell data, we employ the **Soft Actor-Critic (SAC)** reinforcement learning framework, effectively addressing the non-differentiable challenges inherent in data-level problems. Through this methodology, we introduce a paradigm that harnesses reinforcement learning to achieve specific biological objectives in single-cell data analysis.

.. image:: _static/Pattern.png
   :alt: Graphical Abstract
   :width: 600px
   :align: center

Key Features
------------

- **SAC-Based Analysis**: Uses Soft Actor-Critic reinforcement learning for lineage branch identification
- **No Prior Knowledge Required**: Identifies branches without requiring predefined starting points or cell subtypes
- **Interactive Web Interface**: Upload data, set parameters, preprocess, and visualize results online
- **Multiple Input Formats**: Supports ``h5ad`` and 10x Genomics formats
- **Flexible Visualization**: Dimensionality reduction plots and heatmaps with export capabilities

Installation
------------

.. code-block:: bash

    pip install scfocus

Quick Start
-----------

.. code-block:: python

    import scanpy as sc
    import scfocus

    # Load and preprocess data
    adata = sc.read_h5ad('your_data.h5ad')
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var.highly_variable]
    sc.pp.pca(adata)
    sc.pp.neighbors(adata, n_neighbors=15)
    sc.tl.umap(adata)

    # Run scFocus analysis
    embedding = adata.obsm['X_umap']
    focus = scfocus.focus(embedding, n=6, pct_samples=0.01)
    focus.meta_focusing(n=3)
    focus.merge_fp2()

    # Add results to AnnData
    adata.obsm['focus_probs'] = focus.mfp[0]

Web Interface
-------------

Launch the interactive web interface:

.. code-block:: bash

    scfocus ui

Or access the hosted version at `scfocus.streamlit.app <https://scfocus.streamlit.app/>`_.

Citation
--------

Chen, C., Fu, Z., Yang, J., Chen, H., Huang, J., Qin, S., Wang, C., & Hu, X. (2025).
scFocus: Detecting Branching Probabilities in Single-cell Data with SAC.
*Computational and Structural Biotechnology Journal*.
`doi:10.1016/j.csbj.2025.04.036 <https://doi.org/10.1016/j.csbj.2025.04.036>`_

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: TUTORIALS

   notebook/HematoAging
   notebook/LungRegeneration
   notebook/NeuronsRehabilatation
   notebook/4Datasets-t-SNE

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: API

   environment
   focus
   model
   utils
   cli
