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
---

# GLIMPSE — Generalized Locality for Scalable and Robust CT

Coordinate-based CT reconstruction from sparse-view sinograms
([paper](https://ieeexplore.ieee.org/abstract/document/11018464),
[arXiv](https://arxiv.org/abs/2401.00816),
[code](https://github.com/swing-research/Glimpse)). GLIMPSE predicts each pixel
from only the sinogram data local to that pixel, which gives strong
out-of-distribution generalization.

## This checkpoint

| | |
|---|---|
| Image size | {image_size} |
| Projection angles (views) | {n_angles} |
| Noise (SNR, dB) | {noise_snr} |
| Forward operator | `{geometry}` (circle={circle}) |

## Usage

```python
from glimpse import GlimpseModel
model = GlimpseModel.from_pretrained("{repo_id}").eval()
# reconstruct: see glimpse.reconstruct.reconstruct_image and the demo notebook
# at https://github.com/swing-research/Glimpse
```

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
    with open(os.path.join(out_dir, 'README.md'), 'w') as f:
        f.write(_CARD.format(
            image_size=config.image_size, n_angles=config.n_angles,
            noise_snr=config.noise_snr, geometry=config.geometry,
            circle=config.circle, repo_id=repo_id))
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
