# Publishing GLIMPSE on the Hugging Face Hub

A walkthrough for putting the paper, model checkpoints, and datasets on the
[🤗 Hub](https://huggingface.co), as suggested in the repo issue.

## 0. One-time setup

```sh
pip install -U huggingface_hub          # already in environment.yml / requirements.txt
huggingface-cli login                   # paste a token from https://huggingface.co/settings/tokens
```

## 1. Paper page

Submit the arXiv paper at <https://huggingface.co/papers/submit> to get a paper
page (discussion + links to artifacts). Once your models/datasets are up, link
them from the paper page and claim the paper on your profile.

## 2. Model checkpoints

`GlimpseModel` subclasses `PyTorchModelHubMixin`, so loading is a one-liner:

```python
from glimpse import GlimpseModel
model = GlimpseModel.from_pretrained("your-username/glimpse-lodopab-128-50").eval()
```

Push a trained checkpoint (HF recommends **one repo per checkpoint** so download
stats work). The helper builds the model from its config, loads the `.pt`
weights, writes `config.json` + `model.safetensors` + a model card, and uploads:

```sh
# preview locally first (no upload)
python scripts/push_to_hub.py --config configs/lodopab.yaml --checkpoint glimpse.pt \
    --local-dir hub_export --local-only

# then push
python scripts/push_to_hub.py --config configs/lodopab.yaml --checkpoint glimpse.pt \
    --repo-id your-username/glimpse-lodopab-128-50
```

Use a descriptive repo id per checkpoint, e.g. encode the key settings
(`glimpse-lodopab-<image_size>-<n_angles>`). The model card (auto-written by the
script) already includes `license: mit`, discovery tags, the checkpoint's
config, a usage snippet, and the citation.

## 3. Datasets

Host the raw image folders so they're discoverable (sinograms are rendered on
the fly, so only the images are stored):

```sh
python scripts/push_dataset_to_hub.py --data-dir datasets \
    --repo-id your-username/lodopab-ct-glimpse
```

This uploads `train/`, `test/`, `ood/` and a dataset card describing the layout.
(For a full `datasets`-library loader with the dataset viewer, a small loading
script could be added later — not required to host the files.)

**Licensing / attribution (important).** "Published" does not mean
"unrestricted" — but in this case redistribution *is* allowed with credit:

- **LoDoPaB-CT** (the CT slices) is released under the **Open Data Commons
  Attribution License (ODC-By) 1.0** — DOI 10.5281/zenodo.3384092 (Leuschner et
  al., *Scientific Data* 2021). ODC-By permits redistribution and reuse **with
  attribution**.
- It is derived from **LIDC-IDRI** (via TCIA), under **CC BY 3.0** — also
  attribution-only.

The **`ood/` brain images** are a separate dataset — the **CT-ICH**
intracranial-hemorrhage scans (Hssayeni et al., *Data* 2020), on PhysioNet
([ct-ich](https://physionet.org/content/ct-ich/), DOI 10.13026/4nae-zg36) under
**CC BY 4.0** — also redistribution-with-attribution.

So both sources are publishable with credit. Because they carry different
licenses (`odc-by` vs `cc-by-4.0`), the cleanest option is **two separate HF
dataset repos** (LoDoPaB CT subset; CT-ICH OOD), each with its own license tag —
rather than one mixed repo. Since both already have canonical homes (Zenodo /
PhysioNet), upload just your *processed subsets* with pointers to the originals,
and check whether copies already exist on the Hub so you can link instead of
duplicate. The provided dataset card includes the full attribution and
citations.

## 4. Link everything

On the paper page, add the GitHub URL and link each model/dataset repo. Add tags
on the model/dataset pages so they surface in
<https://huggingface.co/models> and <https://huggingface.co/datasets> filters.

> Note: the bundled `glimpse.pt` is trained with `configs/lodopab.yaml`
> (ODL geometry, `circle=false`). Always push a checkpoint together with the
> config it was trained with so `from_pretrained` rebuilds the matching model.
