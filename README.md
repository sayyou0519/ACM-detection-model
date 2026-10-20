# ACM-detection-model
Random Forest-Based Hyperspectral Classification Pipeline

This repository contains code for pixel-wise inference on hyperspectral imagery using a pretrained Random Forest (RF) model. It provides inference and visualization utilities, including spectral band selection, RGB visualization, class overlays, and GeoTIFF/PNG export.

## Associated publication

This repository is associated with the following published research:

> Hyeji Sim, Jaehyung Yu, Lei Wang, Chanhyeok Park, Huy Hoa Huynh, and Hyun-Cheol Kim, “Case Adaptive Detection Models for Asbestos-Containing Building Materials Based on Hyperspectral Image Processing,” *IEEE Transactions on Geoscience and Remote Sensing*, vol. 63, article no. 5533822, 2025. https://doi.org/10.1109/TGRS.2025.3636112

The paper was published in 2025. This repository provides public Random Forest inference code associated with the research. It does not contain or claim to reproduce the complete model training, experimental validation, or all analyses reported in the paper.

### How to cite

If you use this repository, please cite the associated paper:

```text
H. Sim, J. Yu, L. Wang, C. Park, H. H. Huynh, and H.-C. Kim, “Case Adaptive Detection Models for Asbestos-Containing Building Materials Based on Hyperspectral Image Processing,” IEEE Transactions on Geoscience and Remote Sensing, vol. 63, Art. no. 5533822, 2025, doi: 10.1109/TGRS.2025.3636112.
```

A machine-readable citation is provided in [CITATION.cff](CITATION.cff).

## Features

- Pixel-wise Random Forest classification
- Band selection for feature extraction
- RGB composite generation using selected spectral bands
- Prediction overlay mask generation
- Grayscale prediction visualization
- High-resolution PNG export
- GeoTIFF export preserving original metadata

## Repository structure

```
├── rf_predict.py                 # Main prediction and visualization script
├── RF_model.joblib               # (Optional) pretrained Random Forest model
├── sample_input.tif              # (Optional) example hyperspectral input
├── outputs/
│    ├── RF_prediction.tif
│    ├── RF_overlay_300dpi.png
│    ├── RF_prediction_gray.png
│    ├── RF_class1_mask.png
└── README.md
```

## Requirements

Install necessary Python packages:

```bash
pip install numpy rasterio joblib matplotlib
```

## How to run

1. Prepare a hyperspectral image in GeoTIFF format and a trained RF model in joblib format.
2. Execute the script:

```bash
python rf_predict.py
```

The script performs image loading, band selection, RF inference, RGB visualization, class overlay creation, GeoTIFF export, and 300 dpi PNG visualization export.

## Output files

| File | Description |
| --- | --- |
| `RF_prediction.tif` | GeoTIFF predicted label map |
| `RF_overlay_300dpi.png` | RGB composite with Class 1 overlay |
| `RF_prediction_gray.png` | Grayscale label map |
| `RF_class1_mask.png` | Custom-colored mask for Class 1 |
