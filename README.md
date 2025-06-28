# VCF\_ML\_filter

## Project Overview

**VCF\_ML\_filter** is a project focused on filtering genetic variant call data (VCF files) using machine learning techniques. The goal is to distinguish true variant calls from false positives or low-quality calls by training models on annotated variant data. The repository contains code for data preprocessing, feature analysis, model training (using algorithms like LightGBM and neural networks), and applying those models to new VCF datasets.&#x20;

## Repository Structure

```
VCF_ML_filter/
├── data/
│   ├── EKETA/                # Annotated VCF from EKETA
│   ├── GIAB/                 # GIAB high-confidence calls (hg19 & hg38)
│   ├── SEQC/                 # SEQC annotated variant calls
│   ├── models/               # Trained models and scalers
│   ├── training/             # VCFs used for training
│   └── validation/           # VCFs used for validation/testing
│
├── src/
│   ├── notebooks/
│   │   ├── lightgbm/         # LightGBM training, SHAP, PCA, prediction
│   │   ├── neural/           # Neural network training and usage
│   │   ├── preprocessing/    # Preprocessing/normalization notebooks
│   │   └── utils/            # Feature analysis, flagging, MI, splits
│   └── python/
│       ├── paths.py          # Centralized file path definitions
│       └── vcf_helpers.py    # Helper functions for VCF handling and ML
│
├── .gitignore
├── LICENSE
└── README.md
```

The repository is organized into two main directories (`data` and `src`), along with some standard configuration files (like `.gitignore` and `LICENSE`). Below is a breakdown of the structure and contents:

### `data/` – Contains input datasets, model files, and prepared data splits for training and validation

* `EKETA/`, `GIAB/`, `SEQC/` – Subfolders with raw VCF files from different sources (e.g., GIAB stands for Genome in a Bottle, and SEQC may refer to a sequencing consortium).&#x20;
* `models/` – Saved machine learning model artifacts and related files. For example, you will find the LightGBM model (`lightgbm_variant_classifier.pkl`) and scalers (`standard_scaler.pkl`, `robust_scaler.pkl`), as well as a `model_metadata.json` describing selected features and thresholds.
* `training/` – Prepared training datasets (combined or annotated VCFs) and derived feature data. This folder includes VCF files (like `seqc_training.vcf`, `training_hg38.vcf`, etc.) and a CSV of feature vectors (`vcf_feature_vectors.csv`) used to train the ML models.
* `validation/` – Prepared validation datasets (held-out VCFs) for evaluating the model’s performance.&#x20;

### `src/` – Contains all code, divided into Jupyter notebooks and Python modules

* `notebooks/` – A collection of Jupyter notebooks organized by topic, used for experiments, data analysis, and model development. The notebooks are grouped into subfolders:

  * `lightgbm/` – Notebooks related to the LightGBM model pipeline. For example, there are notebooks for mutual information analysis with SHAP values (`mi_lightgbm_shap.ipynb`), applying PCA to features (`mi_pca_lightgbm.ipynb`), and using the LightGBM model to predict/filter variants (`predict_with_lightgbm.ipynb`).
  * `neural/` – Notebooks for the neural network approach. These include training a neural network on the variant data (`nn.ipynb`) and using the trained network to make predictions (`predict_with_nn.ipynb`).
  * `preprocessing/` – Notebooks dealing with data preprocessing steps. They likely cover normalization and combination of datasets.
  * `utils/` – Notebooks for various utility analyses and tasks. (These are described in detail in a separate section below.)

* `python/` – A small Python module containing helper scripts used by the notebooks and potentially for command-line use. It currently includes:

  * `paths.py` – Defines and manages file system paths used throughout the project.&#x20;
  * `vcf_helpers.py` – A utility module with functions to handle VCF data. This  includes functions to parse VCF files and transform VCF entries into feature vectors for the ML model.&#x20;

## Python Utility Scripts (`src/python`)

Under the `src/python` directory, the repository provides two key Python scripts that act as utilities for the project’s codebase:

### `paths.py`

This script centralizes important file path definitions. Rather than scattering file system paths throughout the code, `paths.py` defines variables or functions that construct paths to datasets, output directories, and model files.  Notebooks can import this module to easily get the correct paths.&#x20;

### `vcf_helpers.py`

This module provides helper functions for working with VCF data. The functions here  handle tasks such as reading VCF files into a usable form (e.g., using `pysam` or `vcfpy` to iterate over records) and extracting relevant features from each variant (like quality scores, depth, allele frequencies, annotations, etc.)

## Utility Notebooks (`src/notebooks/utils`)

The `src/notebooks/utils` directory contains a set of Jupyter notebooks that serve helping roles in the project. These “utility” notebooks are not directly about building a specific model, but rather about understanding the data and setting up the experiments. Here’s what each of these notebooks is for:

### `compare_feature_distributions.ipynb`

This notebook is used to compare the distributions of features across different datasets or groups of variants. In a project where data comes from multiple sources (like GIAB vs. SEQC, or different sequencing runs), it’s crucial to see if their feature distributions differ.&#x20;

### `flag_vcf.ipynb`

This notebook takes a VCF file and a golden set (i.e., a set of known high-confidence variants) and flags the variants found in the VCF as `GOLDEN`. These labels are then used as target values during training and validation of the machine learning models. It does not perform prediction, but rather prepares supervised learning labels by intersecting the input VCF with the reference truth set.

### `mutual_information.ipynb`

This notebook  calculates mutual information for the features in the dataset with respect to the target labels (true variant vs. false variant). Mutual information is a measure from information theory that tells you how much knowing a feature reduces uncertainty about the outcome.

### `validation_split.ipynb`

As the name suggests, this notebook handles splitting the data into training and validation sets. Given a pool of annotated variant data, the notebook  performs a stratified split to create the training dataset and the validation (or testing) dataset.&#x20;
