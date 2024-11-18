import streamlit as st
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import joblib
import io 
import tempfile
from utils import read_files, preprocess, run_umap, run_tsne, run_focus

st.markdown("<h1 style='text-align: center;'>scFocus 🔍 </h1>", unsafe_allow_html=True)  

if 'adata' not in st.session_state:  
    st.session_state.adata = None  
if 'embedding' not in st.session_state:
    st.session_state.embedding = None
if 'processed' not in st.session_state:  
    st.session_state.processed = False  

if not st.session_state.adata:
    uploaded_files = st.sidebar.file_uploader("📁 Sequencing files", accept_multiple_files=True)
    if len(uploaded_files) < 1:
        st.warning("Please upload your files first!", icon="⚠")
    else:
        st.session_state.adata = read_files(uploaded_files)  

if st.session_state.adata is not None:

    n_top_genes = st.sidebar.number_input(
        "Number of highly variable genes",
        min_value=200, max_value=5000, value=2000, step=100
    )
    n_neighbors = st.sidebar.number_input(
        "Number of neighbors",
        min_value=2, max_value=50, value=15, step=1
    )
    min_dist = st.sidebar.number_input(
        "Minimum distance",
        min_value=0.0, max_value=2.0, value=0.5, step=.1
    )
    n_branch = st.sidebar.number_input(
        "Number of branch",
        min_value=2, max_value=10, value=6, step=1
    )
    
    
    if st.sidebar.button("Process"):
        preprocess(st.session_state.adata, n_top_genes)
        st.success("Preprocessing completed!")
        st.session_state.embedding = run_umap(st.session_state.adata, n_neighbors, min_dist)
        st.success("UMAP embedding completed!")
        st.write(st.session_state.adata)
        
        mfp = run_focus(st.session_state.embedding, n_branch)
        st.session_state.adata.obsm['mfp'] = mfp
        for i in range(mfp.shape[1]):
            st.session_state.adata.obs[f'Fate_{i}'] = mfp[:,i]
        st.session_state.processed = True
            
    if st.session_state.processed:
    
        adata = st.session_state.adata
        with tempfile.NamedTemporaryFile(delete=False, suffix='.h5ad') as tmp_file:
            adata.write_h5ad(tmp_file.name)
            tmp_file.seek(0)
            buffer = io.BytesIO()
            with open(tmp_file.name,'rb')as f:
                buffer.write(f.read())
        buffer.seek(0)
        st.sidebar.download_button(label="Download scFocus Processed Data",data=buffer,file_name='adata.h5ad',mime='application/x-h5ad')

        color = st.selectbox('Please select the observation for coloring', options=list(st.session_state.adata.obs))
        sc.pl.umap(st.session_state.adata, color=color, show=False)
        fig = plt.gcf()
        st.pyplot(fig)

        if st.sidebar.button("FateProbs"):
            sc.set_figure_params()
            mfp = st.session_state.adata.obsm['mfp']
            sc.pl.umap(st.session_state.adata, color=[f'Fate_{i}' for i in range(mfp.shape[1])], show=False)
            fig = plt.gcf()
            st.pyplot(fig)
        
        if st.sidebar.button("Heatmap"):
            mfp = st.session_state.adata.obsm['mfp']
            focus_labs = np.argmax(mfp, axis=1)
            st.session_state.adata.obs['focus_labels'] = focus_labs.astype(str)
            with st.spinner("⏳Heatmap run..."):
                mfp1 = mfp[np.argsort(np.argmax(mfp, axis=1)),:]
                container = []
                idxs = []
                labels=[]
                
                for i in range(mfp1.shape[1]):
                    idx = np.where(np.argmax(mfp1, axis=1) == i)[0]
                    idx1 = np.argsort(mfp1[idx, i])[::-1] + idx.min()
                    idxs.append(idx.min())
                    container.append(mfp1[idx1, :])
                    labels.extend([str(i)] * len(idx1)) 
                mfp2 = np.vstack(container)
                fig = plt.figure(figsize=(5,5), dpi=300)
                ax = sns.heatmap(mfp2, yticklabels=False, vmax=1, vmin=0, cmap='viridis')
                ax.set_xticklabels([f'Fate_{i}' for i in range(mfp2.shape[1])])
                st.pyplot(fig)
        
     
        

