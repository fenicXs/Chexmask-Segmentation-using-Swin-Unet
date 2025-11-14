# CheXmask SwinSegmentation Pipeline

This repository packages a minimal end-to-end workflow for preparing chest X-ray segmentation data with the CheXmask dataset utilities and training Swin Transformer based models for downstream experiments. Only the assets required to reproduce our preprocessing and modeling steps are retained, keeping the footprint light while preserving transparency.

## Upstream Projects

- **CheXmask-Database** – Dataset preparation utilities, preprocessing scripts, and metadata definitions for the CheXmask multi-dataset segmentation benchmark. Original repository: [ngaggion/CheXmask-Database](https://github.com/ngaggion/CheXmask-Database).
- **Swin-Unet** – Official PyTorch implementation of Swin Transformer U-Net variants for medical image segmentation. Original repository: [HuCaoFighting/Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet).

Both projects were instrumental in shaping this workflow. Please refer to their licenses and citations when reusing the code or dataset products contained here.

## Repository Layout

The repository intentionally mirrors the minimum structure needed for our experiments:

- `MIMIC files/` – Official metadata releases from the MIMIC-CXR dataset (filenames, split information, CheXpert label outputs). These tables are required before running CheXmask preprocessing pipelines.
- `SwinSegmentation/` – Adapted training, inference, and utility scripts for Swin Transformer segmentation models. This folder consolidates code sourced from Swin-Unet along with lightweight adjustments for our runs.
- `VinDR/` – Precomputed train/validation splits for VinDr-CXR studies used during segmentation benchmarking.
- `chexpert/CheXpert-v1.0_train.csv` and `chexpert/CheXpert-v1.0_valid.csv` – Label tables from the CheXpert dataset required by both CheXmask preprocessing and Swin-Unet training procedures.

All other upstream assets (e.g., model checkpoints, large raw datasets, visualization notebooks) are intentionally omitted to keep the repository focused.

## Getting Started

1. **Clone the repository**
   ```powershell
   git clone <your-fork-url>
   cd <repository-folder>
   ```
2. **Create a Python environment** compatible with PyTorch (CUDA optional) and install dependencies referenced in `SwinSegmentation/requirements.txt`.
3. **Acquire imaging data**
   - Follow the instructions in the CheXmask-Database README to request and download DICOM/JPEG images for MIMIC-CXR, CheXpert, or VinDr datasets. The metadata tables provided here assume the official releases.
4. **Preprocess masks and annotations**
   - Use the scripts from the original CheXmask-Database repository, pointing them to the metadata in `MIMIC files/` and the CheXpert label CSVs included in this repository. Generated masks or derived artifacts should be stored outside of version control.
5. **Train Swin-based models**
   - Prepare dataset lists or dataloaders using `SwinSegmentation/utils/dataset.py`.
   - Launch experiments with `SwinSegmentation/train.py`. Refer to `SwinSegmentation/config.py` for hyperparameter settings and dataset paths.
6. **Run inference or evaluation** using `SwinSegmentation/inference.py`, providing model checkpoints and dataset descriptors aligned with your preprocessing outputs.

## Notes on Data Usage

- The CheXmask-derived masks remain governed by the CheXmask-Database license. Ensure compliance with the usage restrictions of each underlying dataset (MIMIC-CXR, CheXpert, VinDr-CXR).
- No raw images are distributed in this repository. Only official metadata tables and curated splits are included.

## Acknowledgements

This work stands on the shoulders of the CheXmask-Database and Swin-Unet teams. Please cite their publications when publishing results obtained from this workflow.
