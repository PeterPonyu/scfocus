import streamlit as st
import scanpy as sc
import scfocus
import os
import tempfile
from io import BytesIO

@st.cache_data
def preprocess(_adata, n_top_genes):
    with st.spinner("Normalizing total counts..."):
        sc.pp.normalize_total(_adata, target_sum=1e4)
        st.success("Normalization completed!")
        
    with st.spinner("Logarithmizing data..."):
        sc.pp.log1p(_adata)
        st.success("Logarithmizing completed!")
        
    with st.spinner("Selecting highly variable genes..."):
        sc.pp.highly_variable_genes(_adata, n_top_genes=int(n_top_genes))
        _adata = _adata[:, _adata.var.highly_variable]
        st.success("Highly variable genes selected!")
        
    with st.spinner("Running PCA..."):
        sc.pp.pca(_adata, mask_var='highly_variable')
        st.success("PCA completed!")
@st.cache_data        
def run_umap(_adata, n_neighbors, min_dist):
    with st.spinner("Computing neighbors..."):
        sc.pp.neighbors(_adata, n_neighbors=int(n_neighbors))
    with st.spinner("Computing UMAP embedding..."):
        sc.tl.umap(_adata, min_dist=min_dist)
    embedding = _adata.obsm['X_umap'].copy()
    return embedding
@st.cache_data    
def run_tsne(_adata, perplexity):
    with st.spinner("Computing t-SNE embedding..."):
        sc.tl.tsne(_adata, perplexity=int(perplexity))
    st.success("t-SNE completed!", icon="🎉")
    embedding = _adata.obsm['X_tsne'].copy()
    return embedding
@st.cache_data    
def run_focus(_embedding, n=6, pct_samples=.01, meta_focusing=3):
    with st.spinner("scFocus running..."):
        focus = scfocus.focus(_embedding, n=n, pct_samples=pct_samples).meta_focusing(n=meta_focusing)
        focus.merge_fp2()
        st.success("scFocus completed!", icon="🎉")
    return focus.mfp[0]


@st.cache_data
def read_files(uploaded_files):
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
    """Read a single uploaded file and return an AnnData object."""
    file_type = uploaded_file.name.rsplit('.', 1)[-1].lower()
    try:
        if file_type == 'h5ad':
            return sc.read_h5ad(BytesIO(uploaded_file.read()))
        elif file_type in ['mtx', 'mtx.gz']:
            return sc.read_mtx(BytesIO(uploaded_file.read()))
        elif file_type == 'loom':
            return sc.read_loom(BytesIO(uploaded_file.read()))
        elif file_type == 'csv':
            return sc.read_csv(BytesIO(uploaded_file.read()))
        elif file_type == 'txt':
            return sc.read_text(BytesIO(uploaded_file.read()))
        elif file_type == 'xlsx':
            return sc.read_excel(BytesIO(uploaded_file.read()))
        elif file_type in ['h5', 'h5ad.gz']:
            return sc.read_10x_h5(BytesIO(uploaded_file.read()))
        else:
            st.error(f"Unsupported file type: `{file_type}`", icon="🤔")
            return None
    except Exception as e:
        st.error(f"Failed to read `{file_type}` file: {e}", icon="🤔")
        return None

def read_10x_files(mtx_file, features_file, barcodes_file):
    """Read 10x Genomics files (compressed or uncompressed) and return an AnnData object."""
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
