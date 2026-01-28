import streamlit as st  
from pathlib import Path 

def main():  
    st.title('About scFocus')  
    
    st.header('Overview')
    st.write('scFocus is an analytical approach that applies reinforcement learning algorithms to single-cell RNA sequencing data. It enables biologically meaningful analyses by identifying distinct lineage branches without requiring prior knowledge of differentiation starting points or cell subtypes.')
    
    st.write('By utilizing branch probabilities, scFocus enhances cell subtype discrimination. The tool employs the Soft Actor-Critic (SAC) reinforcement learning framework, addressing non-differentiable challenges inherent in data-level problems.')
    
    st.header('Features')
    st.write('This web interface allows researchers to perform data preprocessing, dimensionality reduction, and visualization. The following features are available:')
    
    st.markdown('''
    1. **Upload data**: Support for h5ad and 10x Genomics formats
    2. **Configure parameters**: Number of highly variable genes, neighbors, minimum distance, and branches
    3. **Run analysis**: Normalization, log transformation, highly variable gene selection, UMAP embedding, and scFocus analysis
    4. **Visualize results**: Dimensionality reduction plots and heatmaps
    5. **Download results**: Export processed data in h5ad format
    ''')
    
    st.header('Graphical Abstract')
    current_dir = Path(__file__).parent  
    image_path = current_dir.parent / 'graphic_abstract.png'  
    st.image(str(image_path), caption='scFocus workflow overview', width=800)

if __name__ == '__main__':  
    main()
