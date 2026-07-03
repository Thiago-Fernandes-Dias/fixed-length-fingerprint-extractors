# AGENTS.md — `flx` (Fixed-Length Fingerprint Representation Extractor)

## Setup

```powershell
pip install -r requirements.txt    # includes `-e .` to install the `flx` package
```

Python 3.9+. The package is the `flx/` directory; edits there are live if installed via `-e`.

## Dev commands

| Task          | Command              |
|---------------|----------------------|
| Run tests     | `pytest`             |
| Format code   | `black .`            |

Tests live in `tests/`. Only `test_datasets.py` exists — it verifies dataset loaders and identifiers. Run `pytest tests/test_datasets.py` for a focused run.

## Project structure

- **`flx/`** — installable package. Entry points for all ML workflows.
  - `flx/data/` — dataset loaders (`ImageLoader` subclasses), embeddings, poses, minutia maps, ISO encoder/decoder.
  - `flx/models/` — DeepPrint model architectures (InceptionV4-based), loss functions, training loop.
  - `flx/extractor/` — fixed-length embedding extraction.
  - `flx/benchmarks/` — verification & identification benchmarks.
  - `flx/setup/` — predefined models, datasets, and experiments. This is the **registry layer**: add new extractors in `experiments.py`, new dataset loaders in `datasets.py`.
  - `flx/scripts/` — runnable entry points: training, extraction, reweighting, benchmarking, plotting.
  - `flx/image_processing/` — augmentation and binarization.
  - `flx/visualization/` — OpenCV-based debugging display.
- **`main2.py`** (root) — standalone benchmark script using `flx`, **not** part of the package. References paths like `./fixed-length-fingerprint-extractors/models` and `datasets/FVC/...` — this script may require path adjustments.
- **`notebooks/`** — Jupyter tutorials for datasets, training, and embedding extraction.
- **`data/`** — input/output data. Only `data/benchmarks/` is tracked in git; fingerprint datasets, embeddings, and poses are gitignored.
- **`models/`** — trained models (gitignored). Structure: `models/<model_name>/best_model.pyt`.

## Key conventions

### Identifiers

Every fingerprint sample is keyed by `Identifier(subject_id, impression)`. "Subject" = distinct finger (not person). "Impression" = single capture. `IdentifierSet` wraps ordered collections of them.

### Dataset naming

Datasets live under `data/fingerprints/<name>/`. Canonical names used in `flx/setup/datasets.py` and `flx/setup/experiments.py`:
- `SFingev2` (train), `SFingev2ValidationSeparateSubjects` (val), `SFingev2TestNone` / `SFingev2TestCapacitive` / `SFingev2TestOptical`
- `mcyt330_optical`, `mcyt330_capacitive`
- `FVC2004_DB1A`
- `NIST SD4`

### Image loading

`ImageLoader` subclasses auto-discover files by filename pattern from a root directory. `TransformedImageLoader` wraps an `ImageLoader` and applies poses + transforms. Input size for DeepPrint is **299×299**.

### Binarization thresholds (per sensor)

Defined in `flx/setup/datasets.py`:

| Sensor              | Threshold |
|---------------------|-----------|
| SFinge (synthetic)  | 5.0       |
| MCYT capacitive     | 4.8       |
| MCYT optical        | 3.8       |
| FVC2004             | 1.8       |
| NIST SD4            | 4.0       |

### Model naming convention

`DeepPrint_[Loc][Tex][Minu]_<NDIMS>` — e.g. `DeepPrint_TexMinu_512`. `Loc` = localization network, `Tex` = texture branch, `Minu` = minutia branch. `NDIMS` = embedding dimensionality. For combined Tex+Minu variants, `NDIMS` is the **per-branch** dim (e.g. `DeepPrint_TexMinu_512` uses 256 dims per branch — see `experiments.py:118`).

### Training

Requires fingerprint datasets (`SFingev2`, `mcyt330_optical`, `mcyt330_capacitive`) under `data/fingerprints/`. Edit `flx/scripts/run_extractor_training.py` to pick model + validation set, then run it. Training uses `get_training_set()` which concatenates 6000 SFinge + 2000 MCYT (both sensors) subjects. GPU (A100-class) recommended; CPU needs 8+ cores for preprocessing throughput.

### Pre-trained models

Downloaded from [Google Drive](https://drive.google.com/drive/folders/1vV2skXApZMhqWTlF2j_qgXDxRYan5U1f). Place `best_model.pyt` into the corresponding `models/<model_name>/` directory.

## Common pitfalls

- `requirements.txt` lists `opencv-python` twice (lines 9–10). Don't duplicate further.
- `main2.py` hardcodes paths that assume a parent directory named `fixed-length-fingerprint-extractors`. Prefer using `flx/scripts/` entry points instead.
- Embeding loader `combine()` concatenates texture and minutia embeddings by ID — both must exist for the same identifier set.
- Validation benchmark files must be pre-generated (JSON) in `data/benchmarks/verification/<testset_name>.json` for training validation to work.
