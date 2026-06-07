"""Push a trained GLIMPSE checkpoint to the Hugging Face Hub.

Builds the model from a config, loads a training checkpoint into it, writes a
model card, and uploads ``config.json`` + ``model.safetensors`` + ``README.md``
to a model repo. Authenticate first with ``huggingface-cli login`` (or set the
``HF_TOKEN`` env var).

Examples
--------
    # dry run: just write the Hub-format files locally to inspect them
    python scripts/push_to_hub.py --config configs/lodopab.yaml \
        --checkpoint glimpse.pt --local-dir hub_export --local-only

    # upload to the Hub (creates the repo if needed)
    python scripts/push_to_hub.py --config configs/lodopab.yaml \
        --checkpoint glimpse.pt --repo-id your-username/glimpse-lodopab-128-50

Afterwards anyone can load it with::

    from glimpse import GlimpseModel
    model = GlimpseModel.from_pretrained("your-username/glimpse-lodopab-128-50")
"""

import argparse
import os
import tempfile

from glimpse import Config, build_model, load_checkpoint

_CARD = """---
license: mit
library_name: glimpse-ct
tags:
- computed-tomography
- inverse-problems
- image-reconstruction
- implicit-neural-representation
- sparse-view-ct
pipeline_tag: image-to-image
{datasets_yaml}---

# GLIMPSE — Generalized Locality for Scalable and Robust CT

Coordinate-based CT reconstruction from sparse-view sinograms
([paper](https://ieeexplore.ieee.org/abstract/document/11018464),
[arXiv](https://arxiv.org/abs/2401.00816),
[code](https://github.com/swing-research/Glimpse)), published in *IEEE Transactions
on Medical Imaging*. Instead of reconstructing a whole image at once, GLIMPSE
predicts **one pixel at a time** from only the sinogram data *local* to that
pixel's coordinate. This locality makes it resolution-agnostic and gives strong
**out-of-distribution generalization** — e.g. train on natural images / faces and
reconstruct medical brain scans without retraining.

## This checkpoint

| | |
|---|---|
| Image size | {image_size} |
| Projection angles (views) | {n_angles} |
| Noise (SNR, dB) | {noise_snr} |
| Forward operator | `{geometry}` (circle={circle}) |
| Parameters | ~1.3 M |

## Results (LoDoPaB-CT, 50 views, calibrated)

GLIMPSE substantially outperforms classical filtered back-projection (FBP), both
in-distribution and out-of-distribution:

| Set | FBP PSNR / SSIM | GLIMPSE PSNR / SSIM |
|---|---|---|
| In-distribution (LoDoPaB-CT test) | 30.8 dB / 0.79 | **38.0 dB / 0.93** |
| Out-of-distribution (brain CT) | 26.1 dB / 0.51 | **31.6 dB / 0.88** |

## Usage

```python
import numpy as np, torch
from glimpse import GlimpseModel, Config
from glimpse.operators import build_operator
from glimpse.reconstruct import make_coordinate_grid, reconstruct_image

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = GlimpseModel.from_pretrained("{repo_id}").eval().to(device)

# Build the matching ODL parallel-beam operator (50 views over [0, 180) deg).
cfg = Config.from_yaml('configs/lodopab.yaml')        # from the GitHub repo
_, init_angles = cfg.resolve_angles()
operator = build_operator(cfg, np.deg2rad(init_angles))

volume = torch.as_tensor(my_image[None], dtype=torch.float32, device=device)  # (1, H, W)
sino = operator.project(volume)                       # sparse-view sinogram
coords = make_coordinate_grid(cfg.image_size).unsqueeze(0).to(device)
recon = reconstruct_image(sino, coords, 1, model, chunk_size=1024)
recon = recon.reshape(cfg.image_size, cfg.image_size)
```

See the [demo notebook](https://github.com/swing-research/Glimpse/blob/main/notebooks/inference_demo.ipynb)
for an end-to-end example (data download, FBP baseline, PSNR/SSIM, figures).

## Citation

```bibtex
@article{{khorashadizadeh2025glimpse,
  title   = {{GLIMPSE: Generalized Locality for Scalable and Robust CT}},
  author  = {{Khorashadizadeh, AmirEhsan and Debarnot, Valentin and Liu, Tianlin and Dokmani{{\\'c}}, Ivan}},
  journal = {{IEEE Transactions on Medical Imaging}},
  year    = {{2025}},
  doi     = {{10.1109/TMI.2025.3568017}}
}}
```
"""


def parse_args():
    p = argparse.ArgumentParser(description="Push a GLIMPSE checkpoint to the HF Hub.")
    p.add_argument('--config', required=True, help='YAML config the checkpoint was trained with.')
    p.add_argument('--checkpoint', default=None, help='Training checkpoint .pt (default: <exp>/glimpse.pt).')
    p.add_argument('--repo-id', default=None, help='Target HF repo, e.g. user/glimpse-lodopab-128-50.')
    p.add_argument('--dataset-repo', default=None,
                   help='Optional HF dataset repo to link in the card metadata, e.g. user/lodopab-ct-glimpse.')
    p.add_argument('--local-dir', default=None, help='Where to write the Hub-format files.')
    p.add_argument('--local-only', action='store_true', help='Only write files locally; do not upload.')
    p.add_argument('--private', action='store_true', help='Create the Hub repo as private.')
    return p.parse_args()


def main():
    args = parse_args()
    config = Config.from_yaml(args.config)
    _, init_angles = config.resolve_angles()
    model = build_model(config, init_angles)

    checkpoint = args.checkpoint or os.path.join(config.exp_path, 'glimpse.pt')
    missing, unexpected = load_checkpoint(model, checkpoint, map_location='cpu')
    print(f'Loaded {checkpoint} (missing={missing}, unexpected={unexpected})')

    out_dir = args.local_dir or tempfile.mkdtemp(prefix='glimpse_hub_')
    model.save_pretrained(out_dir)
    repo_id = args.repo_id or '<your-username>/glimpse'
    datasets_yaml = f'datasets:\n- {args.dataset_repo}\n' if args.dataset_repo else ''
    with open(os.path.join(out_dir, 'README.md'), 'w') as f:
        f.write(_CARD.format(
            image_size=config.image_size, n_angles=config.n_angles,
            noise_snr=config.noise_snr, geometry=config.geometry,
            circle=config.circle, repo_id=repo_id, datasets_yaml=datasets_yaml))
    print(f'Wrote Hub-format model to {out_dir}')

    if args.local_only or not args.repo_id:
        print('Local-only (or no --repo-id): not uploading. '
              'Authenticate with `huggingface-cli login` and pass --repo-id to push.')
        return

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(args.repo_id, repo_type='model', private=args.private, exist_ok=True)
    api.upload_folder(repo_id=args.repo_id, repo_type='model', folder_path=out_dir)
    print(f'Pushed to https://huggingface.co/{args.repo_id}')


if __name__ == '__main__':
    main()
