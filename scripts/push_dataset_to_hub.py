"""Upload the GLIMPSE raw-image datasets to a Hugging Face dataset repo.

The published datasets are raw images (``.npy`` CT slices, ``.jpg`` brain
images); sinograms are rendered on the fly. This uploads the extracted folders
as-is to a HF dataset repo so they're hosted and discoverable, and writes a
dataset card describing the layout. Authenticate first with
``huggingface-cli login`` (or set ``HF_TOKEN``).

Example
-------
    python scripts/push_dataset_to_hub.py --data-dir datasets \
        --repo-id your-username/lodopab-ct-glimpse
"""

import argparse
import os
import tempfile

_CARD = """---
license: odc-by
pretty_name: LoDoPaB-CT (GLIMPSE subsets)
tags:
- computed-tomography
- medical-imaging
- inverse-problems
task_categories:
- image-to-image
---

# LoDoPaB-CT subsets for GLIMPSE

Processed image subsets used by
[GLIMPSE](https://github.com/swing-research/Glimpse)
([paper](https://arxiv.org/abs/2401.00816)). Sinograms are **not** stored; they
are rendered on the fly by the GLIMPSE data pipeline (ODL / scikit-image).

## Layout

| Split | Contents | Format |
|---|---|---|
| `train/` | LoDoPaB-CT training slices | `.npy` float32 arrays |
| `test/`  | LoDoPaB-CT test slices | `.npy` float32 arrays |
| `ood/`   | out-of-distribution brain images | `.jpg` |

```python
import numpy as np  # a .npy slice
img = np.load("train/0.npy")        # (H, W) float32 in [0, 1]
```

## License & attribution

This repo mixes two sources with different (but both attribution-only) licenses;
credit both when reusing.

**`train/` + `test/` — LoDoPaB-CT** (Leuschner et al., *Scientific Data* 2021),
**ODC-By v1.0**, DOI [10.5281/zenodo.3384092](https://doi.org/10.5281/zenodo.3384092).
Built on **LIDC-IDRI** from [TCIA](https://www.cancerimagingarchive.net/collection/lidc-idri/)
(**CC BY 3.0**). Slices resized/curated for GLIMPSE; originals at the Zenodo DOI.

**`ood/` — CT-ICH intracranial-hemorrhage scans** (Hssayeni et al., *Data* 2020),
**CC BY 4.0**, PhysioNet [ct-ich](https://physionet.org/content/ct-ich/),
DOI 10.13026/4nae-zg36.

> Tip: for clean license tagging you may prefer two separate HF dataset repos
> (LoDoPaB subset `odc-by`; CT-ICH OOD `cc-by-4.0`).

```bibtex
@article{leuschner2021lodopabct,
  title   = {LoDoPaB-CT, a benchmark dataset for low-dose computed tomography reconstruction},
  author  = {Leuschner, Johannes and Schmidt, Maximilian and Baguer, Daniel Otero and Maass, Peter},
  journal = {Scientific Data}, volume = {8}, number = {1}, pages = {109}, year = {2021}
}
@article{hssayeni2020ctich,
  title   = {Intracranial Hemorrhage Segmentation Using a Deep Convolutional Model},
  author  = {Hssayeni, Murtadha and Croock, Muayad and Salman, Aymen and Al-khafaji, Hassan and Yahya, Zakaria and Ghoraani, Behnaz},
  journal = {Data}, volume = {5}, number = {1}, pages = {14}, year = {2020}
}
```
"""


def parse_args():
    p = argparse.ArgumentParser(description="Push GLIMPSE raw datasets to the HF Hub.")
    p.add_argument('--data-dir', default='datasets', help='Folder containing train/ test/ ood/.')
    p.add_argument('--repo-id', required=True, help='Target dataset repo, e.g. user/lodopab-ct-glimpse.')
    p.add_argument('--private', action='store_true', help='Create the dataset repo as private.')
    p.add_argument('--allow-patterns', nargs='*', default=['*/*.npy', '*/*.jpg', '*/*.png'],
                   help='Glob patterns of files to upload (skips junk).')
    return p.parse_args()


def main():
    args = parse_args()
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(args.repo_id, repo_type='dataset', private=args.private, exist_ok=True)

    # Upload the dataset card.
    with tempfile.TemporaryDirectory() as d:
        card = os.path.join(d, 'README.md')
        with open(card, 'w') as f:
            f.write(_CARD)
        api.upload_file(path_or_fileobj=card, path_in_repo='README.md',
                        repo_id=args.repo_id, repo_type='dataset')

    # Upload the raw image folders.
    api.upload_folder(
        repo_id=args.repo_id, repo_type='dataset', folder_path=args.data_dir,
        allow_patterns=args.allow_patterns)
    print(f'Pushed dataset to https://huggingface.co/datasets/{args.repo_id}')


if __name__ == '__main__':
    main()
