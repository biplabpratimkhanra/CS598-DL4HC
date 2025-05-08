# CS598MedFuse

Multi-modal Fusion of Clinical Time-Series and Chest X-ray Imaging using LSTM-Based Architectures

This repository contains our re-implementation and extension of the MedFuse framework for ICU outcome prediction, focusing on LSTM-based time-series encoders in a late-fusion pipeline with chest X-ray image features.

## Table of Contents

- [Background](#background)
- [Features](#features)
- [Requirements](#requirements)
- [Getting Started](#getting-started)
  - [1. Clone the Repo](#1-clone-the-repo)
  - [2. Setup Environment](#2-setup-environment)
  - [3. Data Access & Preparation](#3-data-access--preparation)
  - [4. Running Experiments](#4-running-experiments)
- [Model Details](#model-details)
- [Results](#results)
- [Directory Structure](#directory-structure)
- [Citation](#citation)
- [License](#license)

---

## Background

MedFuse (Hayat et al., 2022) introduced a late-fusion LSTM model to handle partially paired EHR time-series and chest X-ray data for ICU mortality prediction.
This project builds on that LSTM-based pipeline and compares fusion strategies on MIMIC-IV & MIMIC-CXR.

---

## Features

- **Time-series encoder:** LSTM (2-layer, hidden=256) for clinical signals.
- **Image encoder:** DenseNet-121 pretrained on ImageNet for chest X-rays.
- **Late-fusion module** to combine modalities while handling missing data.
- **Benchmark tasks:**
  - In-hospital mortality prediction (binary)
  - Phenotype classification (25 labels)
- **Fusion strategies:** concatenation vs. attention-based weighting.
- **Reproducible** scripts with fixed seeds, logging, and evaluation metrics (AUROC, AUPRC).

---

## Requirements

- Python 3.8+
- PyTorch 1.10+
- torchvision
- numpy, pandas, scikit-learn
- tqdm, pyyaml, wandb (optional)
- CUDA 11.x (for GPU training)

Install dependencies via:

```bash
pip install -r requirements.txt
```

---

## Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/biplabpratimkhanra/CS598-DL4HC.git
cd CS598-DL4HC/CS598MedFuse
```

### 2. Setup Environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

### 3. Data Access & Preparation

1. **MIMIC-IV** clinical CSVs & **MIMIC-CXR** images require PhysioNet credentialing.
2. Place extracted CSV files in `data/mimiciv/` and DICOM/JPEG images in `data/mimiccxr/`.
3. Run preprocessing:

   ```bash
   python scripts/preprocess_timeseries.py      --input_dir data/mimiciv      --output_dir processed/timeseries      --sampling_rate 2h

   python scripts/preprocess_images.py      --input_dir data/mimiccxr      --output_dir processed/images      --format jpg
   ```

### 4. Running Experiments

Train the LSTM-based MedFuse model:

```bash
python train.py   --encoder lstm   --fusion concat   --task mortality   --data_dir processed   --output_dir runs/lstm_concat_mort
```

Run with attention-based fusion:

```bash
python train.py   --encoder lstm   --fusion attention   --task phenotyping   --data_dir processed   --output_dir runs/lstm_attention_pheno
```

Evaluate saved checkpoints:

```bash
python evaluate.py   --checkpoint runs/lstm_attention_pheno/best.ckpt   --data_dir processed   --metrics auroc auprc
```

---

## Model Details

- **Time-Series Encoder**
  - `lstm`: 2-layer, hidden size 256

- **Image Encoder**
  - DenseNet-121 pretrained on ImageNet, output 1024-dim.

- **Fusion**
  - **Concatenate** clinical + image features -> classifier
  - **Attention**-based learnable weighting of modalities.

---

## Results

| Task                         | Fusion    | AUROC  | AUPRC  |
|------------------------------|-----------|--------|--------|
| In-hospital mortality (all)  | Concat    | 0.758  | 0.413  |
| In-hospital mortality (all)  | Attention | 0.867  | 0.537  |
| Phenotype (mean over 25)     | Concat    | 0.758  | 0.413  |
| Phenotype (mean over 25)     | Attention | 0.867  | 0.537  |

*Full detailed results in `reports/`.*

---

## Directory Structure

```
CS598MedFuse/
├── data/                     # raw & preprocessed data
├── models/                   # saved checkpoints
├── scripts/                  # preprocessing & utility scripts
├── src/
│   ├── encoders/             # LSTM encoder implementation
│   ├── fusion/               # fusion modules
│   ├── datasets/             # PyTorch Dataset & Dataloader
│   └── train.py              # training entrypoint
│   └── evaluate.py           # evaluation & metrics
├── requirements.txt
├── README.md                 # ← this file
└── reports/                  # plots, tables, writeups
```

---

## Citation

If you use this code or data, please cite our work and the original MedFuse paper:

> Hayat, N., Geras, K.J. (2022). _MedFuse: Multi-modal fusion with clinical time-series data and chest X-ray images_. arXiv preprint arXiv:2207.07027.  
> Singh, G. & Khanra, B. (2025). _CS598MedFuse: LSTM-based late-fusion of clinical time-series and chest X-ray data_. GitHub repository. https://github.com/biplabpratimkhanra/CS598-DL4HC

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
