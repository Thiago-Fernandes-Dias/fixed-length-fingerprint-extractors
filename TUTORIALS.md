# flx: Module Usage Tutorials

This document provides a comprehensive guide on how to use the modules in the `flx` package, synthesizing the workflows from the interactive notebooks in `notebooks/` and covering the newly integrated FLARE representation pipeline.

---

## Table of Contents

1. [Data Management (`flx.data`)](#1-data-management-flxdata)
   - [1.1 Identifiers & Identifier Sets](#11-identifiers--identifier-sets)
   - [1.2 Custom DataLoaders](#12-custom-dataloaders)
   - [1.3 Datasets (Indexing, Zipping, Concatenation)](#13-datasets-indexing-zipping-concatenation)
   - [1.4 Loading Fingerprints from Disk](#14-loading-fingerprints-from-disk)
   - [1.5 Image Preprocessing & Augmentation](#15-image-preprocessing--augmentation)
2. [Model Training with DeepPrint (`flx.models` & `flx.extractor`)](#2-model-training-with-deepprint-flxmodels--flxextractor)
   - [2.1 Model Instantiation](#21-model-instantiation)
   - [2.2 Preparing Training Data](#22-preparing-training-data)
   - [2.3 Running `extractor.fit()`](#23-running-extractorfit)
3. [DeepPrint Extraction & Benchmarking](#3-deepprint-extraction--benchmarking)
   - [3.1 Loading Weights & Extracting Embeddings](#31-loading-weights--extracting-embeddings)
   - [3.2 Running Verification Benchmarks](#32-running-verification-benchmarks)
   - [3.3 Evaluating Metrics & Plotting DET Curves](#33-evaluating-metrics--plotting-det-curves)
4. [FLARE Pipeline (`flx.extractor.flare` & `flx.benchmarks.matchers`)](#4-flare-pipeline-flxextractorflare--flxbenchmarksbiometric_comparison)
   - [4.1 Official FLARE Full Pipeline (4-Combination)](#41-official-flare-full-pipeline-4-combination)
   - [4.2 Standalone FDD Descriptor Extraction](#42-standalone-fdd-descriptor-extraction)
   - [4.3 Pose Alignment & Enhancement Modules](#43-pose-alignment--enhancement-modules)
   - [4.4 Matching with `FLAREMatcher`](#44-matching-with-flarematcher)
   - [4.5 Gallery vs. Query Verification Benchmarks](#45-gallery-vs-query-verification-benchmarks)
5. [Quick Reference & Cheatsheet](#5-quick-reference--cheatsheet)

---

## 1. Data Management (`flx.data`)

`flx` cleanly separates **what is in a dataset** (identity tracking) from **how data is fetched from disk or memory** (data loading).

### 1.1 Identifiers & Identifier Sets

An `Identifier` represents a specific sample in a biometric database, composed of:
* `subject: int`: Differentiates distinct fingers / individuals.
* `impression: int`: Differentiates the captures / impressions taken from that specific finger.

```python
from flx.data.dataset import Identifier, IdentifierSet

# Instantiate individual identifiers (0-indexed internally)
sample_a = Identifier(subject=0, impression=1)
sample_b = Identifier(subject=0, impression=2)

# IdentifierSet maintains uniqueness and sorted order by (subject, impression)
id_set = IdentifierSet([sample_a, sample_b, Identifier(subject=1, impression=0)])

print(f"Total samples: {len(id_set)}")
print(f"Total distinct subjects: {id_set.num_subjects}")

# Filtering and set operations
subset = id_set.filter_by_index([0, 1])
is_superset = id_set >= subset  # True
```

### 1.2 Custom DataLoaders

A `DataLoader` loads a value (image, minutiae map, label, or feature vector) given an `Identifier`. To create a custom loader, inherit from `DataLoader` and implement the `get(identifier)` method:

```python
from flx.data.dataset import DataLoader, Identifier

class DictionaryDataLoader(DataLoader):
    def __init__(self, data_dict: dict[Identifier, str]):
        self._data = data_dict

    def get(self, identifier: Identifier) -> str:
        return self._data[identifier]
```

### 1.3 Datasets (Indexing, Zipping, Concatenation)

A `Dataset` combines an `IdentifierSet` and a `DataLoader`. It supports indexing, standard Python iteration, and PyTorch `torch.utils.data.DataLoader` wrapping:

```python
from flx.data.dataset import Dataset

dataset = Dataset(loader=my_loader, ids=id_set)

# Access by index or by identifier
item_0 = dataset[0]
item_by_id = dataset.get(sample_a)

# Zip multiple datasets together (e.g. image + minutia map)
zipped_dataset = Dataset.zip(image_dataset, minutia_dataset)

# Concatenate datasets
# share_subjects=True keeps subject IDs unified across datasets
concatenated = Dataset.concatenate(dataset_1, dataset_2, share_subjects=True)
```

### 1.4 Loading Fingerprints from Disk

The `ImageLoader` base class indexes a directory tree and parses filenames into `Identifier` objects. `flx` includes pre-built loaders:

* `DirectoryImageLoader`: General directory loader for files named `<subject>_<impression>.<ext>` (1-indexed).
* `FVC2004Loader`: Specialized loader for FVC datasets with `.tif` extension.
* `SFingeLoader`: Specialized loader for synthetic SFinGe images with `.png` extension.

```python
from flx.data.image_loader import DirectoryImageLoader
from flx.data.dataset import Dataset

# Load a directory of FVC or custom images
loader = DirectoryImageLoader(root_dir="/path/to/dataset", extension=".tif")
dataset = Dataset(loader=loader, ids=loader.ids)

print(f"Loaded {len(dataset)} images from {dataset.ids.num_subjects} subjects.")
```

To implement a custom directory layout, inherit from `ImageLoader`:

```python
import cv2
import torch
import torchvision.transforms.functional as VTF
from flx.data.dataset import Identifier
from flx.data.image_loader import ImageLoader

class CustomFolderImageLoader(ImageLoader):
    @staticmethod
    def _extension() -> str:
        return ".png"

    @staticmethod
    def _file_to_id_fun(subdir: str, filename: str) -> Identifier:
        # Example pattern: sub_001_imp_02.png
        parts = filename.replace(".png", "").split("_")
        subject_id = int(parts[1]) - 1
        impression_id = int(parts[3]) - 1
        return Identifier(subject_id, impression_id)

    @staticmethod
    def _load_image(filepath: str) -> torch.Tensor:
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        return VTF.to_tensor(img)
```

### 1.5 Image Preprocessing & Augmentation

Use `TransformedImageLoader` to chain pose transformations and image processing functions (e.g. Gabor filtering / binarization):

```python
from flx.data.transformed_image_loader import TransformedImageLoader
from flx.data.image_helpers import pad_and_resize_to_deepprint_input_size
from flx.image_processing.binarization import LazilyAllocatedBinarizer
from flx.image_processing.augmentation import RandomPoseTransform

# Training preprocessing with random pose augmentation and Gabor filtering
training_loader = TransformedImageLoader(
    images=DirectoryImageLoader("/path/to/raw_images", extension=".tif"),
    poses=RandomPoseTransform(),  # Random rotation and translation
    transforms=[
        LazilyAllocatedBinarizer(ridge_width=5.0),
        pad_and_resize_to_deepprint_input_size,
    ],
)
```

For FLARE normalization, use `flare_image_transform`:

```python
from flx.data.image_helpers import flare_image_transform

flare_loader = TransformedImageLoader(
    images=DirectoryImageLoader("/path/to/raw_images", extension=".tif"),
    transforms=[flare_image_transform],  # Affine normalization: scale 0.5 (256/512)
)
```

---

## 2. Model Training with DeepPrint (`flx.models` & `flx.extractor`)

### 2.1 Model Instantiation

Factory functions in `flx.extractor.fixed_length_extractor` instantiate DeepPrint variants:

* `get_DeepPrint_Tex`: Texture-only branch.
* `get_DeepPrint_Minu`: Minutiae-only branch.
* `get_DeepPrint_TexMinu`: Dual texture and minutiae branches.
* `get_DeepPrint_LocTexMinu`: Full pipeline including localization network.

```python
from flx.extractor.fixed_length_extractor import get_DeepPrint_TexMinu, DeepPrintExtractor

# num_dims is the dimensionality per branch (e.g., 256 for texture + 256 for minutiae = 512 total)
extractor: DeepPrintExtractor = get_DeepPrint_TexMinu(
    num_training_subjects=100,
    num_dims=256
)
```

### 2.2 Preparing Training Data

Training requires:
1. `fingerprints`: `Dataset` yielding preprocessed image tensors (`[1, 299, 299]`).
2. `labels`: `Dataset` mapping subjects to integer class indices (`[0..num_subjects - 1]`).
3. `minutia_maps`: `Dataset` yielding minutiae target tensors (for minutiae branches).

```python
from flx.data.dataset import Dataset, IdentifierSet, Identifier
from flx.data.label_index import LabelIndex
from flx.data.minutia_map_loader import SFingeMinutiaMapLoader

training_ids = IdentifierSet([Identifier(s, i) for s in range(100) for i in range(8)])

image_dataset = Dataset(image_loader, training_ids)
label_dataset = Dataset(LabelIndex(training_ids), training_ids)
minutia_dataset = Dataset(SFingeMinutiaMapLoader("/path/to/minutiae"), training_ids)
```

### 2.3 Running `extractor.fit()`

Execute training with early stopping and model checkpointing:

```python
extractor.fit(
    fingerprints=image_dataset,
    minutia_maps=minutia_dataset,
    labels=label_dataset,
    validation_fingerprints=None,
    validation_benchmark=None,
    num_epochs=50,
    out_dir="./saved_models/deepprint_model",
)
```

---

## 3. DeepPrint Extraction & Benchmarking

### 3.1 Loading Weights & Extracting Embeddings

```python
from flx.extractor.fixed_length_extractor import get_DeepPrint_TexMinu
from flx.data.dataset import Dataset

# Instantiate extractor matching trained configuration
extractor = get_DeepPrint_TexMinu(num_training_subjects=8000, num_dims=256)
extractor.load_model("pretrained_models/deepprint/deepprint_texminu_512.pyt")

# Extract fixed-length representations
dataset = Dataset(image_loader, image_loader.ids)
texture_embeddings, minutia_embeddings = extractor.extract(dataset)
```

### 3.2 Running Verification Benchmarks

Combine feature embeddings and execute 1:1 biometric comparisons:

```python
from flx.data.embedding_loader import EmbeddingLoader
from flx.benchmarks.matchers import CosineSimilarityMatcher
from flx.scripts.generate_benchmarks import create_verification_benchmark

# Combine branches into a unified 512-dimensional vector per impression
combined_embeddings = EmbeddingLoader.combine(texture_embeddings, minutia_embeddings)
matcher = CosineSimilarityMatcher(combined_embeddings)

# Generate all genuine and impostor comparison pairs
benchmark = create_verification_benchmark(
    subjects=list(range(dataset.num_subjects)),
    impressions_per_subject=list(range(8)),
)

results = benchmark.run(matcher)
```

### 3.3 Evaluating Metrics & Plotting DET Curves

```python
from flx.visualization.plot_DET_curve import plot_verification_results

print(f"Equal Error Rate (EER): {results.get_equal_error_rate():.4f}")
print(f"FMR100: {results.get_fmr100():.4f}")
print(f"FMR1000: {results.get_fmr1000():.4f}")

# Plot Detection Error Tradeoff (DET) curve
plot_verification_results(
    figure_path="results/deepprint_det",
    results=[results],
    model_labels=["DeepPrint_TexMinu_512"],
    plot_title="Verification Performance on FVC2004",
)
```

---

## 4. FLARE Pipeline (`flx.extractor.flare` & `flx.benchmarks.matchers`)

FLARE extracts fixed-length dense representations (FDD) and supports multi-combination enhancement and pose alignment.

### 4.1 Official FLARE Full Pipeline (4-Combination)

The official FLARE pipeline produces 4 representations per fingerprint by combining 2 pose estimators and 2 image enhancers:
$$\text{2 Poses (Voting, Regression)} \times \text{2 Enhancers (UNetEnh, PriorEnh)} = \text{4 Combinations}$$

```python
from flx.extractor.flare import FLAREFullPipeline
from flx.data.dataset import Dataset
from flx.data.image_loader import DirectoryImageLoader

# Initialize pipeline with pre-trained checkpoints
pipeline = FLAREFullPipeline(
    desc_model_path="pretrained_models/flare/desc/desc_model.pth.tar",
    voting_pose_path="pretrained_models/flare/pose/VotingPose.pth",
    regression_pose_path="pretrained_models/flare/pose/RegressionPose.pth",
    priorenh_dir="pretrained_models/flare/enhancement/priorenh",
    unetenh_path="pretrained_models/flare/enhancement/unetenh/unetenh.pth",
    tar_shape=(256, 256),
    middle_shape=(512, 512),
    device="cuda",
    batch_size=32,
)

# Load dataset
loader = DirectoryImageLoader("/path/to/FVC2000/Db1_a/tif", extension=".tif")
dataset = Dataset(loader, loader.ids)

# Returns a FLAREEmbeddingLoader storing 4 feature vectors [4, 3072] and masks [4, 256] per print
flare_embeddings = pipeline.extract(dataset)
```

### 4.2 Standalone FDD Descriptor Extraction

When running without pose estimators or enhancers (or using custom pre-aligned images):

```python
from flx.extractor.flare import FLAREExtractor

extractor = FLAREExtractor(
    model_path="pretrained_models/flare/desc/desc_model.pth.tar",
    tar_shape=(256, 256),
    middle_shape=(512, 512),
    ndim_feat=6,
    input_norm=False,
    device="cuda",
)

embeddings = extractor.extract(dataset)
```

### 4.3 Pose Alignment & Enhancement Modules

Individual components can be used modularly:

```python
import torch
from flx.extractor.flare import FLAREPoseEstimator
from flx.extractor.enhancement import FLAREEnhancer
from flx.data.image_helpers import flare_image_transform

# 1. Pose Estimation
pose_estimator = FLAREPoseEstimator(
    pose_type="VotingPose",  # or "RegressionPose"
    model_path="pretrained_models/flare/pose/VotingPose.pth",
)
pose = pose_estimator.predict_pose(raw_img_tensor)  # returns [x, y, theta]

# 2. Affine Alignment using estimated pose
aligned_img = flare_image_transform(raw_img_tensor, pose_2d=pose)

# 3. Enhancement
enhancer = FLAREEnhancer(
    method="UNetEnh",  # or "PriorEnh"
    model_path="pretrained_models/flare/enhancement/unetenh/unetenh.pth",
)
enhanced_tensor = enhancer.enhance_tensor(aligned_img)
```

### 4.4 Matching with `FLAREMatcher`

`FLAREMatcher` implements masked cosine similarity. For multi-combination representations, it computes scores across all matching pairs and takes the **maximum score**:

```python
from flx.benchmarks.matchers import FLAREMatcher

matcher = FLAREMatcher(flare_embeddings)

# Compute similarity score between two samples
sample_1 = dataset.ids[0]
sample_2 = dataset.ids[1]

score = matcher.similarity(sample_1, sample_2)
print(f"Similarity Score: {score:.4f}")
```

### 4.5 Gallery vs. Query Verification Benchmarks

Verification benchmarks split impressions into gallery (reference) and query (search) sets:

```python
from flx.scripts.generate_benchmarks import create_verification_gallery_query_benchmark

# Split impressions: first 4 as gallery, next 4 as query
benchmark = create_verification_gallery_query_benchmark(
    subjects=list(range(100)),
    gallery_impressions=list(range(0, 4)),  # impressions 1 to 4
    query_impressions=list(range(4, 8)),    # impressions 5 to 8
)

results = benchmark.run(matcher)

# Export full score matrix to Parquet or CSV
results.to_parquet("results/flare_verification_results.parquet")
# or: results.to_csv("results/flare_verification_results.csv")
print(f"FLARE EER: {results.get_equal_error_rate():.4f}")
```

---

## 5. Quick Reference & Cheatsheet

| Task | Key Classes / Functions | Module |
| :--- | :--- | :--- |
| **Track biometric identity** | `Identifier`, `IdentifierSet` | `flx.data.dataset` |
| **Manage data loaders** | `DataLoader`, `Dataset` | `flx.data.dataset` |
| **Load directory of images** | `DirectoryImageLoader`, `FVC2004Loader` | `flx.data.image_loader` |
| **Apply preprocessing/augmentations** | `TransformedImageLoader`, `LazilyAllocatedBinarizer` | `flx.data.transformed_image_loader`, `flx.image_processing` |
| **FLARE image normalization** | `flare_image_transform` | `flx.data.image_helpers` |
| **Train DeepPrint** | `DeepPrintExtractor.fit` | `flx.extractor.fixed_length_extractor` |
| **Extract DeepPrint** | `DeepPrintExtractor.extract` | `flx.extractor.fixed_length_extractor` |
| **Extract official FLARE (4-comb)** | `FLAREFullPipeline.extract` | `flx.extractor.flare` |
| **Extract raw FDD descriptor** | `FLAREExtractor.extract` | `flx.extractor.flare` |
| **Pose estimation** | `FLAREPoseEstimator` | `flx.extractor.flare` |
| **Fingerprint enhancement** | `FLAREEnhancer` | `flx.extractor.enhancement` |
| **Cosine matching (DeepPrint)** | `CosineSimilarityMatcher` | `flx.benchmarks.matchers` |
| **Masked matching (FLARE)** | `FLAREMatcher` | `flx.benchmarks.matchers` |
| **Standard 1:1 Benchmark** | `create_verification_benchmark` | `flx.scripts.generate_benchmarks` |
| **Gallery/Query Benchmark** | `create_verification_gallery_query_benchmark` | `flx.scripts.generate_benchmarks` |
