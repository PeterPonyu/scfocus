# scFocus  

## **Abstract**

Single-cell transcriptomics captures cell differentiation trajectories through changes in gene expression intensity. However, it is challenging to obtain precise information on the composition of gene sets corresponding to each lineage branch in complex biological systems. The combination of branch probabilities and unsupervised clustering can effectively characterize changes in gene expression intensity, reflecting continuous cell states without relying on prior information. In this study, we propose a analytic algorithm named single-cell (sc)-Focus that divides cell subpopulations based on reinforcement learning and unsupervised branching in low-dimensional latent space of single cells. The lineage component strength of scFocus coincides with the expression regions of hallmark genes, capturing differentiation processes more effectively in comparison to the original low-dimensional latent space and showing a stronger subpopulation discriminative power. Furthermore, scFocus is applied to ten single-cell datasets, including small-scale datasets, common-scale datasets, and multi-batch datasets. This demonstrates its applicability on different types of datasets and showcases its potential in discovering biological changes due to experimental treatments through multi-batch dataset processing. Finally, an online analysis tool based on scFocus was developed, helping researchers and clinicians in the process and visualization of single-cell RNA sequencing data as well as the interpretation of these data through branch probabilities in a streamlined and intuitive way.

## **Graphical Abstract**
<p align="center">  
  <img src="source/_static/Pattern.png" alt="Pattern Image" width="600"/>  
</p>

## **Installation**

[![PyPI](https://img.shields.io/pypi/v/scfocus.svg?color=brightgreen&style=flat)](https://pypi.org/project/scfocus/)

``` bash
pip install scfocus
```

## **Documentation**

[![Documentation Status](https://readthedocs.org/projects/scfocus/badge/?version=latest)](https://scfocus.readthedocs.io/en/latest/?badge=latest)

The usage of `scfocus` in notebooks can be found in the [documentation](https://scfocus.readthedocs.io/en/latest/).

## **Streamlit UI**

You can find example data in the data folder.

Access the Web UI by [streamlit cloud app](https://scfocus.streamlit.app/).

Command line tool is also available on linux.

```bash
scfocus ui
```

## **License**
<p>
    <a href="https://choosealicense.com/licenses/mit/" target="_blank">
        <img alt="license" src="https://img.shields.io/github/license/PeterPonyu/scfocus?style=flat-square&color=brightgreen"/>
    </a>
</p>


## **Cite**

- Chen, C., Fu, Z., Yang, J., Chen, H., Huang, J., Qin, S., Wang, C., & Hu, X. (2025). **scFocus: Detecting Branching Probabilities in Single-cell Data with SAC**. *Computational and Structural Biotechnology Journal*. [https://doi.org/10.1016/j.csbj.2025.04.036](https://doi.org/10.1016/j.csbj.2025.04.036)

