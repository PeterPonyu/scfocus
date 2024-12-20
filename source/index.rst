scFocus documentation
=====================

scFocus🔍  
=========  

About scFocus  
-------------  

💗 **scFocus** is an innovative approach that leverages reinforcement learning algorithms to conduct biologically meaningful analyses. By utilizing branch probabilities, scFocus enhances cell subtype discrimination without requiring prior knowledge of differentiation starting points or cell subtypes.  

To identify distinct lineage branches within single-cell data, we employ the **Soft Actor-Critic (SAC)** reinforcement learning framework, effectively addressing the non-differentiable challenges inherent in data-level problems. Through this methodology, we introduce a paradigm that harnesses reinforcement learning to achieve specific biological objectives in single-cell data analysis.  

Features  
--------  

💗 We have developed an interactive website for **scFocus**, designed to help researchers easily perform data preprocessing, dimensionality reduction, and visualization. You can do the following:  

1. **Upload Your Single-Cell Data**  
   - Supports formats including `h5ad` and `10x`.  

2. **Set Parameters**  
   - Configure settings such as:  
     - Number of highly variable genes  
     - Number of neighbors  
     - Minimum distance  
     - Number of branches  

3. **Perform Preprocessing and Dimensionality Reduction Online**  
   - Processes include:  
     - Normalization  
     - Logarithmizing  
     - Highly variable genes selection  
     - Preprocessing  
     - UMAP embedding  
     - scFocus analysis  

4. **Choose Your Visualization Method**  
   - Options include:  
     - Dimensionality reduction plots  
     - Heatmaps  
   - Download the processed files for further analysis.  

.. image:: _static/Pattern.png  
   :alt: Pattern Image  
   :width: 600px  
   :align: center  

   
Installation
------------

.. code-block:: python

    pip install scfocus

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

