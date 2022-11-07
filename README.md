# <p align="center">LOD-Brain</p>


Visit the [Project page](https://rocknroll87q.github.io/LOD-Brain/).

## Description

Implementation of the paper "Fighting the scanner effect in brain MRI segmentation with a progressive level-of-detail network trained on multi-site data" ([link](**** TODO ****)).

<!--<p align="center">
<img src="misc/training.png" width="700" />  
</p>
-->
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

```
@misc{Svanera2022,
  doi = {10.48550/ARXIV.2211.02400},
  url = {https://arxiv.org/abs/2211.02400},
  author = {Svanera, Michele and Savardi, Mattia and Signoroni, Alberto and Benini, Sergio and Muckli, Lars},
  keywords = {3D segmentation, brain MRI, progressive level-of-detail architecture, 3D multi-site learning},
  title = {Fighting the scanner effect in brain MRI segmentation with a progressive level-of-detail network trained on multi-site data},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
