from pathlib import Path

import streamlit as st

def main():
    st.title('About scFocus🔍')
    st.write('💗 scFocus is an innovative approach that leverages reinforcement learning algorithms to conduct biologically meaningful analyses.')
    st.write('By utilizing branch probabilities, scFocus enhances cell subtype discrimination without requiring prior knowledge of differentiation starting points or cell subtypes.')
    st.write('To identify distinct lineage branches within single-cell data, we employ the Soft Actor-Critic (SAC) reinforcement learning framework, effectively addressing the non-differentiable challenges inherent in data-level problems.')
    st.write('💗 Here, we have developed an interactive website for scFocus, designed to help researchers easily perform data preprocessing, dimensionality reduction, and visualization. You can:')
    st.write('1️⃣ Upload your single-cell data for processing online (supporting formats include h5ad, 10x).')
    st.write('2️⃣ Set parameters (Number of highly variable genes, Number of neighbors, Minimum distance, Number of branches).')
    st.write('3️⃣ Perform preprocessing and dimensionality reduction online (Normalization - Logarithmizing - Highly variable genes selection - PCA - UMAP embedding - scFocus analysis).')
    st.write('4️⃣ Choose your visualization method (dimensionality reduction plot, heatmap) and download the processed files.')
    st.write('💗 Why dost thy eyes with radiant splendor shine? Because my focus ever stays on thine.💐')
    
    current_dir = Path(__file__).parent
    image_path = current_dir.parent / 'graphic_abstract.png'
    st.image(str(image_path), caption='An illustrative image of scFocus', use_column_width=True)
    st.markdown("<h1 style='text-align: center;'>🦊 Hope you enjoy using scFocus </h1>", unsafe_allow_html=True)  

if __name__ == '__main__':  
    main()
