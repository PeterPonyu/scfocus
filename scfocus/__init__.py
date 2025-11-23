"""  
scFocus: Single-cell Reinforcement Learning for Lineage Focusing  

scFocus is a Python package for analyzing single-cell RNA sequencing data using  
reinforcement learning techniques. It identifies distinct lineage branches and cell  
fate probabilities without requiring prior knowledge of cell types or differentiation  
starting points.  

The package implements the Soft Actor-Critic (SAC) reinforcement learning framework  
to enhance cell subtype discrimination and discover biologically meaningful patterns  
in single-cell data.  

Main Classes  
------------  
focus : Main class for performing scFocus analysis  
    Implements the reinforcement learning-based approach for identifying cell lineages.  

Functions  
---------  
main : Command-line interface entry point  
    Provides CLI access to scFocus functionality.  

Examples  
--------  
Basic usage with UMAP embedding:  

>>> import scanpy as sc  
>>> import scfocus  
>>> # Load and preprocess data  
>>> adata = sc.read_h5ad('data.h5ad')  
>>> sc.pp.normalize_total(adata, target_sum=1e4)  
>>> sc.pp.log1p(adata)  
>>> sc.pp.pca(adata)  
>>> sc.pp.neighbors(adata)  
>>> sc.tl.umap(adata)  
>>> # Run scFocus  
>>> embedding = adata.obsm['X_umap']  
>>> focus_obj = scfocus.focus(embedding, n=6)  
>>> focus_obj.meta_focusing(n=3)  
>>> focus_obj.merge_fp2()  
>>> # Add results to AnnData  
>>> adata.obsm['focus_probs'] = focus_obj.mfp[0]  

See Also  
--------  
scanpy : Single-cell analysis in Python  
torch : PyTorch deep learning framework  

Notes  
-----  
For detailed documentation and tutorials, visit:  
https://scfocus.readthedocs.io/  

Citation  
--------  
Chen, C., Fu, Z., Yang, J., Chen, H., Huang, J., Qin, S., Wang, C., & Hu, X. (2025).  
scFocus: Detecting Branching Probabilities in Single-cell Data with SAC.  
Computational and Structural Biotechnology Journal.  
https://doi.org/10.1016/j.csbj.2025.04.036  
"""

from .focus import focus
from .cli import main

__all__ = ['focus', 'main']

__version__ = '0.0.4'
