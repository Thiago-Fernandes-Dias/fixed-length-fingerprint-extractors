# flx: Fixed-Length Fingerprint Representation Extractors

Benchmarking and evaluation framework for Deep Learning-based fixed-length fingerprint representation extractors across different embedding sizes, sensor types, and alignment/enhancement pipelines.

This repository integrates two state-of-the-art fixed-length fingerprint representation architectures:
1. **DeepPrint** — Texture- and minutiae-based fixed-length fingerprint extractor (Engelsma et al., IEEE TPAMI 2021; benchmarked in Rohwedder et al., BIOSIG 2023).
2. **FLARE** — Fixed-Length Dense Fingerprint Representation with dual-strategy pose-aware alignment (Voting & Regression) and dual-model enhancement (UNetEnh & PriorEnh) (Pan et al., IEEE TIFS 2026 / WIFS 2024).

## Installation & Setup

The framework requires **Python 3.9+** and PyTorch.

1. Clone the repository and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Install `flx` in editable mode:
   ```bash
   pip install -e .
   ```

For detailed guides and code examples for each module, refer to [TUTORIALS.md](TUTORIALS.md).

---

## 📥 Pre-trained Models & Directory Layout

Pre-trained weights for both DeepPrint and FLARE should be placed inside the `pretrained_models/` directory according to the following layout:

```text
pretrained_models/
├── deepprint/
│   └── deepprint_texminu_512.pyt        # DeepPrint (Texture + Minutiae, 512-dim)
└── flare/
    ├── desc/
    │   └── desc_model.pth.tar           # FDD Dense Descriptor Extractor
    ├── pose/
    │   ├── VotingPose.pth               # Voting-based Pose Estimator (GRIDNET4)
    │   └── RegressionPose.pth           # Regression-based Pose Estimator (Single)
    └── enhancement/
        ├── unetenh/
        │   └── unetenh.pth              # UNet-based Enhancement (SqueezeUNet)
        └── priorenh/
            ├── Prior.ckpt               # Prior codebook checkpoint
            ├── priorenh.pth             # Prior-guided Enhancement (VQFPEnhancer)
            └── vq.yaml                  # PriorEnh configuration
```

### Download Links

| Model | Architecture / Purpose | File Name | Destination Path | Download Link |
| :--- | :--- | :--- | :--- | :--- |
| **DeepPrint (512)** | Texture + Minutiae representation | `deepprint_texminu_512.pyt` | `pretrained_models/deepprint/` | [Google Drive Folder](https://drive.google.com/drive/folders/1vV2skXApZMhqWTlF2j_qgXDxRYan5U1f?usp=drive_link) |
| **FLARE FDD** | Fixed-length Dense Descriptor (FDD) | `desc_model.pth.tar` | `pretrained_models/flare/desc/` | [Google Drive File](https://drive.google.com/file/d/1zvAI57L0TDC7q6kQgNh5_DwSbicjJ4hs/view?usp=drive_link) |
| **VotingPose** | Voting-based alignment network | `VotingPose.pth` | `pretrained_models/flare/pose/` | [Google Drive File](https://drive.google.com/file/d/1Zg4duNJ8mg-fkTACTzpPPK7DgNb9NRvA/view?usp=drive_link) |
| **RegressionPose** | Regression-based alignment network | `RegressionPose.pth` | `pretrained_models/flare/pose/` | [Google Drive File](https://drive.google.com/file/d/1AXpN8GBSqhlIXDilqPLfZf9n4Dc0pEpj/view?usp=drive_link) |
| **UNetEnh** | SqueezeUNet fingerprint enhancer | `unetenh.pth` | `pretrained_models/flare/enhancement/unetenh/` | [Google Drive File](https://drive.google.com/file/d/1U0uP8XoxWc90IPlEGnt2KIe0ATAA2VKl/view?usp=drive_link) |
| **PriorEnh** | Prior-guided ridge enhancer | `priorenh.pth` | `pretrained_models/flare/enhancement/priorenh/` | [Google Drive File](https://drive.google.com/file/d/1h3JD6ZhS_TUaCmBhINqKZ0Pb1ad-FRao/view?usp=drive_link) |
| **Prior Codebook** | Ridge prior latent codebook | `Prior.ckpt` | `pretrained_models/flare/enhancement/priorenh/` | [Google Drive File](https://drive.google.com/file/d/14c0A4qRo_lrqa83e-_UpBvu79qEGkK5q/view?usp=drive_link) |

> **Note:** The scripts also accept checkpoints located directly under `../FLARE/model_weights/` and `../FLARE_ENH/pretrained_model/` when experimenting with adjacent clones of the official repositories.

---

## Model Architectures

### 1. DeepPrint (BIOSIG 2023 / TPAMI 2021)

![DeepPrint Architecture](figures/conceptual-overview-deepprint.png)

* **Localization Network**: Localizes the fingerprint center and crops it to a normalized canvas ($299 \times 299$).
* **Inception v4 Stem**: Shared deep convolutional backbone extracting dense textural and spatial representations.
* **Texture Branch**: Produces a fixed-length textural embedding (e.g., 256 or 512 dimensions).
* **Minutiae Branch**: Multi-task branch predicting minutiae locations/orientations via minutia maps and producing a fixed-length minutiae embedding.
* **Matcher**: Cosine similarity over the combined texture and minutiae embeddings (`CosineSimilarityMatcher`).

### 2. FLARE (IEEE TIFS 2026 / WIFS 2024)

FLARE provides a dense, fixed-length descriptor (FDD) combined with multi-path pose alignment and multi-path image enhancement:

1. **Dual Pose Estimation (Alignment)**:
   * `VotingPose` (`GRIDNET4`): Multi-scale voting network estimating core point $(x, y)$ and rotation angle $\theta$.
   * `RegressionPose` (`FingerPose_2D_Single`): Direct classification-to-vector regression for $(x, y, \theta)$.
2. **Dual Image Enhancement**:
   * `UNetEnh` (`SqueezeUNet`): Direct pixel-to-pixel ridge structure enhancement.
   * `PriorEnh` (`VQFPEnhancer_PCNN`): Vector-quantized prior network leveraging high-quality codebooks for latent and noisy prints.
3. **FDRN Descriptor Extractor (`FDD`)**:
   * Takes normalized $256 \times 256$ inputs (derived from a $512 \times 512$ coordinate frame, scale $= 0.5$).
   * Dual-branch ResNet backbone extracting texture features (`embedding_t`) and minutiae features (`embedding`), along with a foreground mask (`mask`).
   * Output: concatenated $12$-channel feature map ($16 \times 16 \times 12 = 3072$ values) and foreground mask ($16 \times 16 = 256$ values).
4. **4-Combination Max Matching (`FLAREMatcher`)**:
   * Generates 4 distinct representations per fingerprint: $(2\text{ poses}) \times (2\text{ enhancers})$.
   * Matches samples using masked cosine similarity (`calculate_flare_score`) across all representation pairs, taking the maximum similarity score.

## Dataset Organization & Image Naming

Datasets must be placed in a directory where image filenames encode the **subject ID** and **impression ID**.

### Expected File Pattern

```text
<subject_id>_<impression_id>.<extension>
```

* **1-indexed numbering**: Filenames use 1-based indexing for subjects and impressions (e.g. `1_1.tif`, `1_2.tif`, ..., `100_8.tif`).
* Supported extensions include `.tif`, `.png`, `.bmp`, `.jpg`.
* The loaders (`DirectoryImageLoader`, `FVC2004Loader`) automatically map these to 0-indexed `Identifier(subject_id - 1, impression_id - 1)` for PyTorch compatibility.

### Example Dataset Directory

```text
/path/to/FVC2000/Db1_a/tif/
├── 1_1.tif      # Subject 1, Impression 1
├── 1_2.tif      # Subject 1, Impression 2
├── ...
├── 1_8.tif      # Subject 1, Impression 8
├── 2_1.tif      # Subject 2, Impression 1
└── 100_8.tif    # Subject 100, Impression 8
```

### Gallery vs. Query Splitting

In the verification benchmarks (`create_verification_gallery_query_benchmark`), impressions are partitioned into:
* **Gallery Impressions**: `range(0, 4)` (first 4 impressions: `_1` to `_4`)
* **Query Impressions**: `range(4, 8)` (remaining 4 impressions: `_5` to `_8`)
* **Subjects**: e.g., `A_SUBJECTS = list(range(100))` for set A (100 subjects) or `B_SUBJECTS = list(range(101, 110))` for set B.

---

## 🚀 Running Experiments

### 1. DeepPrint Benchmark (`main_deepprint.py`)

Runs the DeepPrint verification benchmark across configured datasets and saves result CSVs:

```bash
python main_deepprint.py
```

**Configuring Datasets in `main_deepprint.py`:**
```python
FOLDERS = [
    ("/path/to/FVC2000/Db1_a/tif", "../results/DeepPrint/fvc_2000_db1_a.csv", A_SUBJECTS),
    ("/path/to/FVC2002/Db1_a/tif", "../results/DeepPrint/fvc_2002_db1_a.csv", A_SUBJECTS),
    ("/path/to/FVC2004/Db1_a/tif", "../results/DeepPrint/fvc_2004_db1_a.csv", A_SUBJECTS),
]
```

### 2. Official FLARE 4-Combination Pipeline (`main_flare.py`)

Executes the complete official FLARE pipeline ($2\text{ Poses} \times 2\text{ Enhancers} = 4\text{ Combinations}$ per fingerprint with max-score matching):

```bash
python main_flare.py
```

**Configuring Datasets in `main_flare.py`:**
```python
FOLDERS = [
    ("/path/to/FVC2000/Db1_a/tif", "../results/FLARE/fvc_2000_db1_a.csv", A_SUBJECTS),
    ("/path/to/FVC2004/Db1_a/tif", "../results/FLARE/fvc_2004_db1_a.csv", A_SUBJECTS),
]
```

**Programmatic Pipeline Usage:**
```python
from flx.extractor.flare import FLAREFullPipeline
from flx.benchmarks.matchers import FLAREMatcher
from flx.data.dataset import Dataset
from flx.data.image_loader import DirectoryImageLoader

pipeline = FLAREFullPipeline(
    desc_model_path="pretrained_models/flare/desc/desc_model.pth.tar",
    voting_pose_path="pretrained_models/flare/pose/VotingPose.pth",
    regression_pose_path="pretrained_models/flare/pose/RegressionPose.pth",
    priorenh_dir="pretrained_models/flare/enhancement/priorenh",
    unetenh_path="pretrained_models/flare/enhancement/unetenh/unetenh.pth",
    device="cuda",
)

loader = DirectoryImageLoader("/path/to/dataset", extension=".tif")
dataset = Dataset(loader, loader.ids)
embeddings = pipeline.extract(dataset)

matcher = FLAREMatcher(embeddings)
score = matcher.similarity(dataset.ids[0], dataset.ids[1])
```

### 3. Single-Combination FLARE Enhancement Benchmark (`main_flare_enh.py`)

Runs a lightweight FLARE pipeline using a single selected pose estimator and enhancement model (e.g. VotingPose + UNetEnh):

```bash
python main_flare_enh.py
```

## Benchmark Results

All benchmark scripts export CSV files into `../results/` containing comprehensive verification metrics:
* Genuine and impostor comparison scores
* False Match Rate (FMR) and False Non-Match Rate (FNMR) curves
* Equal Error Rate (EER) and FMR100 / FMR1000 operating points

## Project Structure

```text
├── LICENSE
├── README.md                      <- Main documentation
├── README.old.md                  <- Original BIOSIG 2023 DeepPrint README
├── TUTORIALS.md                   <- Step-by-step module usage tutorials
├── requirements.txt               <- Environment dependencies
├── setup.py                       <- Packaging script (pip install -e .)
├── main_deepprint.py              <- DeepPrint benchmark entrypoint
├── main_flare.py                  <- Official FLARE full pipeline entrypoint
├── main_flare_enh.py              <- FLARE single-enhancement pipeline entrypoint
├── figures/                       <- Explanatory figures and diagrams
├── notebooks/                     <- Interactive tutorial notebooks
├── tests/                         <- Unit and integration tests
├── pretrained_models/             <- Checkpoint directory for DeepPrint and FLARE
│   ├── deepprint/
│   └── flare/
│       ├── desc/
│       ├── pose/
│       └── enhancement/
├── data/
│   ├── benchmarks/                <- Verification and identification benchmark JSONs
│   ├── embeddings/                <- Saved feature embeddings
│   └── poses/                     <- Precomputed ground-truth poses
└── flx/                           <- Core Python package
    ├── benchmarks/                <- Verification, identification, and matchers (FLAREMatcher, CosineSimilarityMatcher)
    ├── data/                      <- Dataset classes, loaders (DirectoryImageLoader, FVC2004Loader, etc.), image helpers
    ├── extractor/                 <- DeepPrint and FLARE extractor interfaces, enhancement runners
    ├── image_processing/          <- Binarization, data augmentations
    ├── models/                    <- Neural network architectures
    │   ├── deep_print_arch.py     <- DeepPrint model variants
    │   ├── enhancement/           <- UNetEnh and PriorEnh network modules
    │   └── flare/                 <- FDD descriptor, VotingPose (GRIDNET4), RegressionPose
    ├── reweighting/               <- Dimension reweighting algorithms
    └── scripts/                   <- Benchmark generator and utility scripts
```

## 📄 Citations

If you use this repository or any of the implemented models in your research, please cite the corresponding papers:

### DeepPrint & BIOSIG 2023 Benchmark
```bibtex
@inproceedings{Rohwedder-FixedLengthFingerprintDNN-BIOSIG-2023,
    author = {T. Rohwedder and D. Osorio-Roig and C. Rathgeb and C. Busch},
    booktitle = {Intl. Conf. of the Biometrics Special Interest Group (BIOSIG)},
    title = {Benchmarking fixed-length Fingerprint Representations across different Embedding Sizes and Sensor Types},
    year = {2023},
    publisher = {IEEE}
}

@article{engelsma2021deepprint,
    author = {Engelsma, Joshua J. and Cao, Kai and Jain, Anil K.},
    journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
    title = {Learning a Fixed-Length Fingerprint Representation}, 
    year = {2021},
    volume = {43},
    number = {6},
    pages = {1981-1997}
}
```

### FLARE Framework (TIFS 2026 / WIFS 2024)
```bibtex
@article{pan2025flare,
    author = {Pan, Zhiyu and Guan, Xiongjun and Duan, Yongjie and Feng, Jianjiang and Zhou, Jie},
    journal = {IEEE Transactions on Information Forensics and Security}, 
    title = {Fixed-Length Dense Fingerprint Representation With Alignment and Robust Enhancement}, 
    year = {2026},
    volume = {21},
    pages = {1751-1765}
}

@inproceedings{pan2024fdd,
    author = {Pan, Zhiyu and Duan, Yongjie and Feng, Jianjiang and Zhou, Jie},
    booktitle = {IEEE International Workshop on Information Forensics and Security (WIFS)}, 
    title = {Fixed-length Dense Descriptor for Efficient Fingerprint Matching}, 
    year = {2024},
    pages = {1-6}
}
```

## License & Acknowledgements

* **DeepPrint modules**: Distributed under the project license in [LICENSE](LICENSE).
* **FLARE & FLARE_ENH modules**: Released for **academic research and educational purposes only**. Commercial use is strictly prohibited.
* Implementation credits:
  * Remi Cadene: Inception v4 PyTorch implementation ([Cadene/pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch))
  * Dong Chengdong & Hang Zhou: ISO-19794-2 encoder/decoder ([DongChengdongHangZhou/iso-19794-2-decoder-encoder](https://github.com/DongChengdongHangZhou/iso-19794-2-decoder-encoder))
  * Zhiyu Pan et al.: Original FLARE & FLARE_ENH implementations ([Yu-Yy/FLARE](https://github.com/Yu-Yy/FLARE), [Yu-Yy/FLARE_ENH](https://github.com/Yu-Yy/FLARE_ENH))
