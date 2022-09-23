---
layout: page
title: <a href="https://rocknroll87q.github.io/LOD-Brain/">LOD-Brain</a>
---

# Abstract

Many clinical and research studies of the human brain call for an accurate automated MRI segmentation.
Despite the advances granted by deep learning techniques, these methods are usually trained with data from single or few MRI sites or vendors. 
This severely limits the model’s ability to generalise well to external data i.e., new volumes from unseen datasets.
Thanks to the increasing availability of open MRI brain data, we aggregate an unprecedented set of 27,000 T1w volumes from 155 acquisition sites, at 1.5 and 3T, from a population spanning from 8 to 90 years old.
To learn from this broad range of variability, as for scanner noise, vendors, brain morphology, etc., we exploit **LOD-Brain**, a 3D progressive level-of-detail (LOD) network.
Coarser levels are responsible to learn a robust brain prior useful for identifying main brain structures and their locations; finer levels, instead, progressively refine the model to handle site-specific intensity distributions, artifacts, and inter-subject anatomical variations.
Thanks to an extremely low number of parameters if compared to other competitors, **LOD-Brain** returns accurate 3D segmentation masks in few seconds.
Extensive tests demonstrate that our method produces superior results with respect to state-of-the-art solutions, without the need of retraining nor fine-tuning when used on external data. 
The easy portability offered by **LOD-Brain** opens the way for large scale application in both research and clinical settings across different healthcare institutions, patient populations, and imaging technology manufacturers.


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

Visit the Github [page](https://rocknroll87q.github.io/LOD-Brain/results#top) for the source code.

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

