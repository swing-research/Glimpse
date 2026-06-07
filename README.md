# GLIMPSE: Generalized Locality for Scalable and Robust CT

[![IEEE TMI](https://img.shields.io/badge/IEEE%20TMI-published-blue)](https://ieeexplore.ieee.org/abstract/document/11018464)
[![Paper](https://img.shields.io/badge/arXiv-2401.00816-red)](https://arxiv.org/abs/2401.00816)
[![PWC](https://img.shields.io/badge/PapersWithCode-report-blue)](https://paperswithcode.com/paper/glimpse-generalized-local-imaging-with-mlps)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/swing-research/Glimpse/blob/main/notebooks/glimpse_colab.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official PyTorch implementation of **GLIMPSE: Generalized Locality for Scalable and Robust CT**, published in
**IEEE Transactions on Medical Imaging** ([paper](https://ieeexplore.ieee.org/abstract/document/11018464)).

GLIMPSE reconstructs a CT image from a **sparse-view sinogram** by predicting **one pixel at a time**
from only the sinogram data *local* to that pixel's coordinate. Because prediction is local and
coordinate-based, the model is resolution-agnostic and generalizes out-of-distribution — e.g. train on
faces / natural images, reconstruct medical brain scans.

<p align="center"><img src="figures/glimpse.jpg" width="1000"></p>

---

## Quickstart

The **recommended** setup uses the ODL forward operator (GPU), which is the default pipeline and matches the
bundled checkpoint.

```sh
# 1. Environment (recommended: ODL + astra GPU operator)
conda env create -f environment-odl.yml && conda activate glimpse-odl
pip install -e .                       # installs the `glimpse` package

# 2. Run the pretrained-model demo (downloads a small dataset automatically)
jupyter notebook notebooks/inference_demo.ipynb

# 3. Or evaluate the bundled checkpoint from the command line
python evaluate.py --config configs/lodopab.yaml --checkpoint glimpse.pt
```

No GPU / can't install astra? A scikit-image (CPU) operator is available — use `environment.yml` and
`configs/skimage.yaml` instead (you'll need to train your own model; see
[Forward operator](#forward-operator-odl-recommended-or-scikit-image)).

The pretrained weights `glimpse.pt` (128×128, 50 views, calibrated, **ODL geometry**) ship with the repo.

## Repository map

```
glimpse/                 # the importable package
  model.py               # GlimpseModel: filter -> local sampling -> MLP
  geometry.py            # patch templates, Fourier filter, coord reflection
  data.py                # RawImageCTDataset: raw images -> sinogram on the fly
  operators.py           # ODL parallel-beam ray transform + FBP
  config.py              # Config dataclass: YAML + CLI overrides
  reconstruct.py         # coordinate grid, full-image reconstruction, FBP baseline
  metrics.py             # PSNR / SSIM
  engine.py              # train() and evaluate() loops
configs/                 # YAML experiment configs (lodopab [ODL, default], skimage, uncalibrated)
train.py                 # thin CLI entry -> glimpse.engine.train
evaluate.py              # evaluate a checkpoint, write figures + metrics
baselines/unet.py        # U-Net FBP-postprocessing baseline
slurm/train.sbatch       # submit training on a SLURM GPU cluster
notebooks/inference_demo.ipynb
glimpse.pt               # pretrained checkpoint
```

## Configuration

All hyperparameters live in a single [`Config`](glimpse/config.py) dataclass, loaded from a YAML file and
overridable on the command line — there is no global mutable state:

```sh
python train.py --config configs/lodopab.yaml
python train.py --config configs/lodopab.yaml --n-angles 60 --epochs 5000 --learn-geometry true
```

Key options:

| Field | Meaning |
|---|---|
| `image_size`, `n_angles`, `noise_snr` | acquisition geometry; must match the dataset |
| `geometry` | forward operator: `odl` (GPU, default/recommended) or `skimage` (CPU) |
| `uncalibrated_type` | `No` (calibrated), `random`, `fixed`, `blind` — how the *told* angles diverge from the *true* ones |
| `learn_geometry` | learn the projection angles / detector geometry (robustness to miscalibration) |
| `learnable_filter`, `filter_name` | learn the FBP-style Fourier filter (init `ramp`, `shepp-logan`, …) |
| `patch_size`, `patch_shape`, `learn_patch` | the learnable local sampling "glimpse" |
| `pixels_per_step`, `steps_per_batch` | per-pixel training: random pixels optimised per gradient step, and steps per image batch |
| `train`, `restore_model` | train vs. evaluate-only; reload checkpoint first |

Every field is documented inline in the YAML files (with the allowed option values), so the configs are
self-explanatory — open [`configs/lodopab.yaml`](configs/lodopab.yaml) to see them all.

## Forward operator: ODL (recommended) or scikit-image

GLIMPSE can simulate sinograms with two interchangeable forward operators, selected by `geometry`:

| `geometry` | Operator | Detector | Sinogram | Needs |
|---|---|---|---|---|
| `odl` **(default, recommended)** | ODL `Parallel2dGeometry` + `astra_cuda` (GPU) | `[-1,1]`, `N` bins | `(B, n_angles, n_det)` | `environment-odl.yml` |
| `skimage` | scikit-image `radon` (CPU) | diagonal, `⌈√2·N⌉` bins | `(B, n_det, n_angles)` | `environment.yml` |

**ODL is recommended**: it's the fast, GPU-accelerated, widely-used tomography operator, and is the geometry
the bundled `glimpse.pt` was trained with. The model's back-projection (`extract_sin`) matches whichever
geometry is set, so reconstruction is correct either way. With `geometry: 'odl'`, sinograms are generated on
the GPU from raw images during training/eval (`glimpse.operators.ParallelBeam2DOperator`, mirroring ODL's
`RayTransform`).

```sh
# ODL (default)
python train.py --config configs/lodopab.yaml

# scikit-image (CPU alternative; environment.yml, no astra needed)
python train.py --config configs/skimage.yaml
```

> The two geometries use different detector layouts, so a checkpoint is tied to the geometry it was trained
> with. The bundled `glimpse.pt` is **ODL**; to use `geometry: 'skimage'` train your own model.

## Datasets

Datasets are hosted on SwitchDrive. `train`/`test` are raw `.npy` CT images and `ood` are brain `.jpg`
images; sinograms are rendered on the fly (the demo notebook downloads these automatically). After
downloading, point `train_path` / `test_path` / `ood_path` in your config at the folders.

```sh
# Small LoDoPaB-CT subsets (used by the demo)
curl -L -o datasets/train.zip https://drive.switch.ch/index.php/s/qMlALcE7AZzUPBh/download
curl -L -o datasets/test.zip  https://drive.switch.ch/index.php/s/fWBUmtZjozwpN9W/download
curl -L -o datasets/ood.zip   https://drive.switch.ch/index.php/s/BQ8Yb8ofjutsEjV/download
# Complete LoDoPaB-CT
curl -L -O -J https://drive.switch.ch/index.php/s/XzMbtHQFrQsLgxC/download
```

| Set | Link | Size |
|---|---|---|
| Small train subset | [download](https://drive.switch.ch/index.php/s/qMlALcE7AZzUPBh) | ~1000 images |
| Small test subset | [download](https://drive.switch.ch/index.php/s/fWBUmtZjozwpN9W) | ~100 images |
| OOD brain images | [download](https://drive.switch.ch/index.php/s/BQ8Yb8ofjutsEjV) | 18 images |
| Complete LoDoPaB-CT | [download](https://drive.switch.ch/index.php/s/XzMbtHQFrQsLgxC) | full |

Sinograms are rendered on the fly from these raw images (no pre-processing step), so just point
`train_path` / `test_path` / `ood_path` at the extracted folders. The same data is also mirrored on the
🤗 Hub at [`AmirEhsan1995/lodopab-ct-glimpse`](https://huggingface.co/datasets/AmirEhsan1995/lodopab-ct-glimpse)
(see [Hugging Face Hub](#hugging-face-hub) for a one-line download).

## Results (small test subset, 50 views, calibrated)

Reproduce with the bundled `glimpse.pt` (ODL geometry) via `notebooks/inference_demo.ipynb` or
`python evaluate.py --config configs/lodopab.yaml --checkpoint glimpse.pt`. GLIMPSE substantially
outperforms the classical FBP baseline on both in-distribution (LoDoPaB-CT) and out-of-distribution (brain
CT) data:

| Set | FBP PSNR / SSIM | GLIMPSE PSNR / SSIM |
|---|---|---|
| In-distribution (test) | 30.8 dB / 0.79 | **38.0 dB / 0.93** |
| Out-of-distribution (brain) | 26.1 dB / 0.51 | **31.6 dB / 0.88** |

See the demo notebook for qualitative FBP / GLIMPSE / GT / error comparisons.

## Uncalibrated sensor geometry

GLIMPSE can reconstruct even when the projection angles it is told are wrong. Set
`uncalibrated_type` to `random` / `fixed` / `blind` and keep `learn_geometry: true`; the model refines the
sensor geometry during training. See [`configs/uncalibrated.yaml`](configs/uncalibrated.yaml).

## Hugging Face Hub

`GlimpseModel` integrates with the [🤗 Hub](https://huggingface.co/docs/hub) via `PyTorchModelHubMixin`, so a
checkpoint can be shared and loaded in one line. The pretrained 50-view model is published at
[`AmirEhsan1995/Glimpse`](https://huggingface.co/AmirEhsan1995/Glimpse):

```python
from glimpse import GlimpseModel
model = GlimpseModel.from_pretrained("AmirEhsan1995/Glimpse").eval()
```

Publish your own trained checkpoint (after `huggingface-cli login`):

```sh
python scripts/push_to_hub.py --config configs/lodopab.yaml --checkpoint glimpse.pt \
    --repo-id your-username/glimpse-lodopab-128-50
# add --local-only to preview the config.json / model.safetensors / README first
```

The datasets are also published on the Hub at
[`AmirEhsan1995/lodopab-ct-glimpse`](https://huggingface.co/datasets/AmirEhsan1995/lodopab-ct-glimpse)
(`train`/`test` LoDoPaB-CT under ODC-By, `ood` brain CT under CC BY 4.0) and can be pulled directly:

```python
from huggingface_hub import snapshot_download
snapshot_download("AmirEhsan1995/lodopab-ct-glimpse", repo_type="dataset", local_dir="datasets")
```

To publish your own copy (after `huggingface-cli login`):

```sh
python scripts/push_dataset_to_hub.py --data-dir datasets --repo-id your-username/lodopab-ct-glimpse
```

See [`HUGGINGFACE.md`](HUGGINGFACE.md) for the full publishing walkthrough (paper page, model & dataset cards).

## Citation

```bibtex
@article{khorashadizadeh2025glimpse,
  title   = {GLIMPSE: Generalized Locality for Scalable and Robust CT},
  author  = {Khorashadizadeh, AmirEhsan and Debarnot, Valentin and Liu, Tianlin and Dokmani{\'c}, Ivan},
  journal = {IEEE Transactions on Medical Imaging},
  year    = {2025},
  doi     = {10.1109/TMI.2025.3568017}
}
```

## License

Released under the [MIT License](LICENSE).
