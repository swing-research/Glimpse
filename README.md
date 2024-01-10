# GLIMPSE: Generalized Local Imaging with MLPs

[![Paper](https://img.shields.io/badge/arxiv-report-red)](https://arxiv.org/abs/2401.00816)
[![PWC](https://img.shields.io/badge/PWC-report-blue)](https://paperswithcode.com/paper/glimpse-generalized-local-imaging-with-mlps)

This repository is the official Pytorch implementation of "GLIMPSE: Generalized Local Imaging with MLPs".


## Requirements
(This code is tested with PyTorch 1.12.1, Python 3.8.3, CUDA 11.6 and cuDNN 7.)
- numpy
- scipy
- matplotlib
- odl
- imageio
- torch==1.12.1
- torchvision=0.13.1
- astra-toolbox

## Installation

Run the following code to install conda environment "environment.yml":
```sh
conda env create -f environment.yml
```


## Experiments
### Train
All arguments for training are explained in config_funknn.py. After specifying your arguments, you can run the following command to train the model:
```sh
python3 train_funknn.py 
```


## Citation
If you find the code useful in your research, please consider citing the paper.

```
@article{khorashadizadeh2024glimpse,
  title={GLIMPSE: Generalized Local Imaging with MLPs},
  author={Khorashadizadeh, AmirEhsan and Debarnot, Valentin and Liu, Tianlin and Dokmani{\'c}, Ivan},
  journal={arXiv preprint arXiv:2401.00816},
  year={2024}
}
```
