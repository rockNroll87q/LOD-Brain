# <p align="center">LOD-Brain</p>


Visit the [Project page](https://rocknroll87q.github.io/LOD-Brain/).

## Description

Implementation of the paper "Fighting the scanner effect in brain MRI segmentation with a progressive level-of-detail network trained on multi-site data" ([link](https://rocknroll87q.github.io/LOD-Brain/)).

<p align="center">
<img src="misc/training.png" width="700" />  
</p>

This repository includes: 

* The trained model in `/trained_model/7im5hf6z/`
* A notebook for testing the model on a new data
* A `src` folder with the code for training a new model

## Usage

Visit the relative [page](https://rocknroll87q.github.io/LOD-Brain/usage) to learn how to use `CEREBRUM-7T` from source code, docker, or singularity.

## Data

Visit the relative [page](https://rocknroll87q.github.io/LOD-Brain/data) for all the information needed about the data.

## Authors

[Michele Svanera](https://github.com/rockNroll87q)
&
[Mattia Savardi](https://github.com/msavardi)


## Citation

If you find this code useful in your research, please consider citing our paper:

Svanera, M., Savardi, M., Signoroni, A., Benini, S., & Muckli, L. (2024). Fighting the scanner effect in brain MRI segmentation with a progressive level-of-detail network trained on multi-site data. Medical Image Analysis, 93, 103090. https://doi.org/10.1016/j.media.2024.103090

```
@article{SVANERA2024103090,
  title = {Fighting the scanner effect in brain MRI segmentation with a progressive level-of-detail network trained on multi-site data},
  journal = {Medical Image Analysis},
  volume = {93},
  pages = {103090},
  year = {2024},
  issn = {1361-8415},
  doi = {https://doi.org/10.1016/j.media.2024.103090},
  url = {https://www.sciencedirect.com/science/article/pii/S136184152400015X},
  author = {Michele Svanera and Mattia Savardi and Alberto Signoroni and Sergio Benini and Lars Muckli},
  keywords = {3D segmentation, Brain MRI, Progressive level-of-detail architecture, Multi-site learning},
}
```
