---
layout: page
title: <a href="https://rocknroll87q.github.io/LOD-Brain/">LOD-Brain</a>
---

# Abstract

Many clinical and research studies of the human brain require an accurate structural MRI segmentation.
While traditional atlas-based methods can be applied to volumes from any acquisition site, recent deep learning algorithms ensure improved accuracy only when tested on data from the same sites exploited in training (i.e., internal data).
The performance degradation experienced on external data (i.e., unseen volumes from unseen sites) is due to the inter-site variabilities in intensity distributions induced by different MR scanner models, acquisition parameters, and unique artefacts.
To mitigate this site-dependency, often referred to as the *scanner effect*, we propose **LOD-Brain**, a 3D progressive level-of-detail (LOD) convolutional neural network able to segment brain data from any site.
Coarser network levels are responsible to learn a robust anatomical prior useful for identifying brain structures and their locations, while finer levels progressively refine the model to handle site-specific intensity distributions and anatomical variations.
We ensure robustness across sites by training the model on an unprecedented rich dataset aggregating data from open repositories: almost 27,000 T1w volumes from around 160 acquisition sites, at 1.5 - 3T, from a population spanning from 8 to 90 years old.  
Extensive tests demonstrate that **LOD-Brain** produces state-of-the-art results, with no significant difference in performance between internal and external sites, and robust to challenging anatomical variations.
Its portability opens the way for large scale application across different healthcare institutions, patient populations, and imaging technology manufacturers. Code, model, and demo are available at the [project website](https://rocknroll87q.github.io/LOD-Brain/).


<p align="center">
<img src="./misc/training.png" width="80%" />  
<figcaption>In a 3D level-of-detail (LOD) network, each level is a CNN. Lower levels, learning from multi-site data, train coarse and site-independent representations of the brain. Training happens in a bottom-up fashion: after convergence of lower levels, superior ones incorporate their learnt spatial context, refining segmentation masks at the fullest scale. At the bottom, we present examples of potential outputs (i.e., reconstructed meshes of outer GM boundary of a testing volume obtained with BrainVoyager) at different network levels.</figcaption>
</p>

<hr>
# DEMO

TODO

<hr>
# Dataset

Visit the relative [page](https://rocknroll87q.github.io/LOD-Brain/dataset) for a better understanding of the datasets used in this work.

<hr>
# Results

Visit the result [page](https://rocknroll87q.github.io/LOD-Brain/results#top) for more results.

<hr>
# Code

Visit the Github [page](https://github.com/rockNroll87q/LOD-Brain/) for the source code.

<hr>
# Usage

Visit the relative [page](https://rocknroll87q.github.io/LOD-Brain/usage) to learn how to use `LOD-Brain` from source code, docker, or singularity.

<hr>
# Citation

If you find this work useful, please consider citing our paper:

```
TO ADD
```


<hr>
# Acknowledgments

This was supported from the EU Horizon 2020 Framework Programme for Research and Innovation under the Specific Grant Agreement No. 945539 (Human Brain Project SGA3). We recognise the priceless contribution made by several openly available MRI datasets: OpenNeuro (https://openneuro.org/), ABCD (https://abcdstudy.org/), Open Science Framework (https://osf.io/), the Human Connectome Project (http://www.humanconnectomeproject.org/), the NIMH Data Archive (https://nda.nih.gov/), the Open Access Series of Imaging Studies (OASIS) (https://www.oasis-brains.org/), Mindboggle101 (https://mindboggle.info/data.html), the evaluation framework for MR Brain Image Segmentation (MRBrainS) (https://mrbrains18.isi.uu.nl/), the the Amsterdam Open MRI Collection (AOMIC) (https://nilab-uva.github.io/AOMIC.github.io/), the Internet Brain Segmentation Repository (IBSR) (https://www.nitrc.org/projects/ibsr), and the great contribution provided by the International Neuroimaging Datasharing Initiative (INDI) (https://fcon_1000.projects.nitrc.org/), with many datasets, including but not limited to, the Nathan Kline Institute-Rockland Sample (NKI-RS) (http://fcon_1000.projects.nitrc.org/indi/enhanced/), the Information eXtraction from Images project (IXI) (https://brain-development.org/ixi-dataset/), the Autism Brain Imaging Data Exchange (ABIDE) (http://fcon_1000.projects.nitrc.org/indi/abide/), and the Attention Deficit Hyperactivity Disorder (ADHD) (https://fcon_1000.projects.nitrc.org/indi/adhd200/).



<hr>

