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
All datasets have been uploaded to SwitchDrive. You can access the complete [LoDoPaB-CT](https://www.nature.com/articles/s41597-021-00893-z) by downloading it from [here](https://drive.switch.ch/index.php/s/XzMbtHQFrQsLgxC). Additionally, we have made available a smaller subset of the LoDoPaB-CT dataset, comprising approximately 1000 [training](https://drive.switch.ch/index.php/s/qMlALcE7AZzUPBh) and 100 [test](https://drive.switch.ch/index.php/s/fWBUmtZjozwpN9W) samples. Moreover, to evaluate model generalization, we have included out-of-distribution (OOD) [brain images](https://drive.switch.ch/index.php/s/BQ8Yb8ofjutsEjV) consisting of 18 samples. These datasets can be downloaded using the following commands:

Complete LoDoPaB-CT:
```sh
curl -O -J https://drive.switch.ch/index.php/s/XzMbtHQFrQsLgxC/download
```

Small LoDoPaB-CT training subset:
```sh
curl -O -J https://drive.switch.ch/index.php/s/qMlALcE7AZzUPBh/download
```

Small LoDoPaB-CT test subset:
```sh
curl -O -J https://drive.switch.ch/index.php/s/fWBUmtZjozwpN9W/download
```

Out-of-didstribution brain images:
```sh
curl -O -J https://drive.switch.ch/index.php/s/BQ8Yb8ofjutsEjV/download
```
After downloading the datasets, please sepcify the training, test and OOD directories in 'config.py' script.

## Experiments
### Training & Inference
All arguments for training are explained in 'config.py'. After specifying your arguments, you can run the following command to train the model:
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

