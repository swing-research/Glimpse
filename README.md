# GLIMPSE: Generalized Local Imaging with MLPs

[![Paper](https://img.shields.io/badge/arxiv-report-red)](https://arxiv.org/abs/2401.00816)
[![PWC](https://img.shields.io/badge/PWC-report-blue)](https://paperswithcode.com/paper/glimpse-generalized-local-imaging-with-mlps)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1f_YvD9WwKHN1NojIOC-HHGXAT4VgQHkz?usp=sharing)

This repository is the official Pytorch implementation of "GLIMPSE: Generalized Local Imaging with MLPs". 

[Colab demo](https://colab.research.google.com/drive/1f_YvD9WwKHN1NojIOC-HHGXAT4VgQHkz?usp=sharing)


<p float="center">
<img src="figures/glimpse.png" width="1000">
</p>


## Requirements
(This code is tested with PyTorch 1.12.1, Python 3.8.3, CUDA 11.6 and cuDNN 7.)
- numpy
- scipy
- matplotlib
- imageio
- torch==1.12.1
- torchvision=0.13.1

## Installation

Run the following code to install conda environment "environment.yml":
```sh
conda env create -f environment.yml
```

## Dataset
All datasets are uploaded to SwitchDrive. You can download the full [LoDoPaB-CT](https://www.nature.com/articles/s41597-021-00893-z) dataset from [here](https://drive.switch.ch/index.php/s/XzMbtHQFrQsLgxC). We also provided a small subset of LoDoPaB-CT which contains around 1000 and 100 [training](https://drive.switch.ch/index.php/s/qMlALcE7AZzUPBh) and [test](https://drive.switch.ch/index.php/s/fWBUmtZjozwpN9W) samples.
All arguments for training are explained in config.py. You can also download the out-of-distribution (OOD) [brain images](https://drive.switch.ch/index.php/s/fWBUmtZjozwpN9W) which containd 18 samples to assess model generalization. The datasets can also be downloaded from the following commands,

Full LoDoPaB-CT:
```sh
curl -O -J https://drive.switch.ch/index.php/s/XzMbtHQFrQsLgxC/download
```

Small training subset:
```sh
curl -O -J https://drive.switch.ch/index.php/s/qMlALcE7AZzUPBh/download
```

Small test subset:
```sh
curl -O -J https://drive.switch.ch/index.php/s/fWBUmtZjozwpN9W/download
```

Out-of-didstribution brain images:
```sh
curl -O -J https://drive.switch.ch/index.php/s/fWBUmtZjozwpN9W/download
```
After downloading the datasets, you need to sepcify the training, test and OOD directories in config.py script.

## Experiments
### Train
All arguments for training are explained in config.py. After specifying your arguments, you can run the following command to train the model:
```sh
python3 train.py 
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

