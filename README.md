# scFocus🔍  

## About scFocus  

💗 **scFocus** is an innovative approach that leverages reinforcement learning algorithms to conduct biologically meaningful analyses. By utilizing branch probabilities, scFocus enhances cell subtype discrimination without requiring prior knowledge of differentiation starting points or cell subtypes.  

To identify distinct lineage branches within single-cell data, we employ the **Soft Actor-Critic (SAC)** reinforcement learning framework, effectively addressing the non-differentiable challenges inherent in data-level problems. Through this methodology, we introduce a paradigm that harnesses reinforcement learning to achieve specific biological objectives in single-cell data analysis.  

## Features  

💗 We have developed an interactive website for **scFocus**, designed to help researchers easily perform data preprocessing, dimensionality reduction, and visualization. You can do the following:  

1. **Upload Your Single-Cell Data**  
   - Supports formats including `h5ad`, `10x`. 

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

<p align="center">  
  <img src="source/_static/Pattern.png" alt="Pattern Image" width="600"/>  
</p>

## **Documentation**

[![Documentation Status](https://readthedocs.org/projects/scfocus/badge/?version=latest)](https://scfocus.readthedocs.io/en/latest/?badge=latest)

[documentation](https://scfocus.readthedocs.io/en/latest/)

## **Installation**

[![PyPI](https://img.shields.io/pypi/v/scfocus.svg?color=brightgreen&style=flat)](https://pypi.org/project/scfocus/)

``` bash
pip install scfocus
```

## **Streamlit UI**

```bash
scfocus ui
```

## **License**
<p>
    <a href="https://choosealicense.com/licenses/mit/" target="_blank">
        <img alt="license" src="https://img.shields.io/github/license/PeterPonyu/scfocus?style=flat-square&color=brightgreen"/>
    </a>
</p>
