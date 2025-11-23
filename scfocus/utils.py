import streamlit as st
import scanpy as sc
import scfocus
import os
import tempfile
from io import BytesIO

@st.cache_data
def preprocess(_adata, n_top_genes):
    """  
    Preprocess single-cell RNA-seq data using scanpy.  
    
    This function performs standard preprocessing steps including count normalization,  
    log transformation, identification of highly variable genes, and PCA.  
    
    Parameters  
    ----------  
    _adata : anndata.AnnData  
        Annotated data matrix with cells as observations and genes as variables.  
        Modified in place.
    n_top_genes : int  
        Number of highly variable genes to identify.  
    
    Notes  
    -----  
    Preprocessing steps:
    1. Total count normalization to 10,000 counts per cell
    2. Log transformation (log1p)
    3. Highly variable gene identification
    4. PCA on highly variable genes
    """
    with st.spinner("Normalizing total counts..."):
        sc.pp.normalize_total(_adata, target_sum=1e4)
        st.success("Normalization completed!")
        
    with st.spinner("Logarithmizing data..."):
        sc.pp.log1p(_adata)
        st.success("Logarithmizing completed!")
        
    with st.spinner("Selecting highly variable genes..."):
        sc.pp.highly_variable_genes(_adata, n_top_genes=int(n_top_genes))
        _adata._inplace_subset_var(_adata.var.highly_variable)
        st.success("Highly variable genes selected!")
        
    with st.spinner("Running PCA..."):
        sc.pp.pca(_adata)
        st.success("PCA completed!")

@st.cache_data        
def run_umap(_adata, n_neighbors, min_dist):
    """  
    Compute UMAP embedding for single-cell data.  
    
    Parameters  
    ----------  
    _adata : anndata.AnnData  
        Preprocessed annotated data matrix.  
    n_neighbors : int  
        Number of neighbors to use in UMAP computation.  
    min_dist : float  
        Minimum distance parameter for UMAP.  
    
    Returns  
    -------  
    embedding : numpy.ndarray  
        2D UMAP embedding coordinates with shape (n_cells, 2).  
    
    Notes  
    -----  
    This function first computes the neighborhood graph and then runs UMAP.  
    """
    with st.spinner("Computing neighbors..."):
        sc.pp.neighbors(_adata, n_neighbors=int(n_neighbors))
        st.success("Neighbors computed!")
    with st.spinner("Computing UMAP embedding..."):
        sc.tl.umap(_adata, min_dist=min_dist)
        st.success("UMAP completed!")
    embedding = _adata.obsm['X_umap'].copy()
    return embedding

@st.cache_data    
def run_tsne(_adata, perplexity):
    """  
    Compute t-SNE embedding for single-cell data.  
    
    Parameters  
    ----------  
    _adata : anndata.AnnData  
        Preprocessed annotated data matrix.  
    perplexity : int  
        Perplexity parameter for t-SNE computation.  
    
    Returns  
    -------  
    embedding : numpy.ndarray  
        2D t-SNE embedding coordinates with shape (n_cells, 2).  
    """
    with st.spinner("Computing t-SNE embedding..."):
        sc.tl.tsne(_adata, perplexity=int(perplexity))
        st.success("t-SNE completed!", icon="🎉")
    embedding = _adata.obsm['X_tsne'].copy()
    return embedding

@st.cache_data    
def run_focus(_embedding, n=6, pct_samples=.01, meta_focusing=3):
    """  
    Run scFocus analysis on embedding data.  
    
    Parameters  
    ----------  
    _embedding : numpy.ndarray  
        2D embedding coordinates (e.g., from UMAP or t-SNE).  
    n : int, optional  
        Number of parallel agents/branches to identify (default: 6).  
    pct_samples : float, optional  
        Percentage of samples to use in each training step (default: 0.01).  
    meta_focusing : int, optional  
        Number of meta-focusing iterations (default: 3).  
    
    Returns  
    -------  
    focus_probs : numpy.ndarray  
        Matrix of focus probabilities with shape (n_cells, n_branches).  
    """
    with st.spinner("scFocus running..."):
        focus = scfocus.focus(_embedding, n=n, pct_samples=pct_samples).meta_focusing(n=meta_focusing)
        focus.merge_fp2()
        st.success("scFocus completed!", icon="🎉")
    return focus.mfp[0]


@st.cache_data
def read_files(uploaded_files):
    """  
    Read uploaded single-cell data files and return an AnnData object.  
    
    Supports multiple file formats:  
    - Single .h5ad file  
    - 10x Genomics format (matrix.mtx, features.tsv, barcodes.tsv)  
    
    Parameters  
    ----------  
    uploaded_files : list  
        List of uploaded file objects from Streamlit file uploader.  
    
    Returns  
    -------  
    adata : anndata.AnnData or None  
        Annotated data matrix if successful, None otherwise.  
    """
    if len(uploaded_files) > 1:
        mtx_file = next((f for f in uploaded_files if 'matrix' in f.name.lower()), None)
        features_file = next((f for f in uploaded_files if 'features' in f.name.lower()), None)
        barcodes_file = next((f for f in uploaded_files if 'barcodes' in f.name.lower()), None)

        if mtx_file and features_file and barcodes_file:
            with st.spinner("Loading 10x Genomics data..."):
                adata = read_10x_files(mtx_file, features_file, barcodes_file)
            if adata is not None:
                st.success("10x Genomics files read successfully! 🎉")
                st.write(adata)
                return adata
        else:
            st.error(
                "Please upload all required 10x Genomics files: "
                "`matrix.mtx`/`matrix.mtx.gz`, `features.tsv`/`features.tsv.gz`, "
                "and `barcodes.tsv`/`barcodes.tsv.gz`.",
                icon="🤔"
            )
    elif len(uploaded_files) == 1:
        with st.spinner("Loading single file..."):
            adata = read_uploaded_file(uploaded_files[0])
        if adata is not None:
            st.success("File read successfully! 🎉")
            st.write(adata)
            return adata
    else:
        st.error("No files uploaded.", icon="🚨")
    return None

def read_uploaded_file(uploaded_file):
    """  
    Read a single uploaded file and return an AnnData object.  
    
    Parameters  
    ----------  
    uploaded_file : UploadedFile  
        Uploaded file object from Streamlit.  
    
    Returns  
    -------  
    adata : anndata.AnnData or None  
        Annotated data matrix if successful, None otherwise.  
    
    Notes  
    -----  
    Currently only supports .h5ad format.
    """
    file_type = uploaded_file.name.rsplit('.', 1)[-1].lower()
    try:
        if file_type == 'h5ad':
            return sc.read_h5ad(BytesIO(uploaded_file.read()))
        else:
            st.error(f"Unsupported file type: `{file_type}`", icon="🤔")
            return None
    except Exception as e:
        st.error(f"Failed to read `{file_type}` file: {e}", icon="🤔")
        return None

def read_10x_files(mtx_file, features_file, barcodes_file):
    """  
    Read 10x Genomics files (compressed or uncompressed) and return an AnnData object.  
    
    Parameters  
    ----------  
    mtx_file : UploadedFile  
        Matrix file (matrix.mtx or matrix.mtx.gz).  
    features_file : UploadedFile  
        Features/genes file (features.tsv or features.tsv.gz).  
    barcodes_file : UploadedFile  
        Barcodes file (barcodes.tsv or barcodes.tsv.gz).  
    
    Returns  
    -------  
    adata : anndata.AnnData or None  
        Annotated data matrix if successful, None otherwise.  
    """
    try:
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Save uploaded files to temporary directory with their original names
            mtx_path = os.path.join(tmpdirname, mtx_file.name)
            features_path = os.path.join(tmpdirname, features_file.name)
            barcodes_path = os.path.join(tmpdirname, barcodes_file.name)

            with open(mtx_path, 'wb') as f:
                f.write(mtx_file.read())
            with open(features_path, 'wb') as f:
                f.write(features_file.read())
            with open(barcodes_path, 'wb') as f:
                f.write(barcodes_file.read())

            # Check for compressed or uncompressed files
            required_files = [
                'matrix.mtx', 'matrix.mtx.gz',
                'features.tsv', 'features.tsv.gz',
                'barcodes.tsv', 'barcodes.tsv.gz'
            ]
            temp_files = os.listdir(tmpdirname)
            for file_variant in required_files:
                if not any(f.startswith(file_variant.split('.')[0]) for f in temp_files):
                    raise ValueError(f"Missing required file: `{file_variant}`")

            # Read the data using Scanpy
            adata = sc.read_10x_mtx(tmpdirname, var_names='gene_symbols', cache=True)
            return adata
    except Exception as e:
        st.error(f"Failed to read 10x Genomics files: {e}", icon="🤔")
        return None
