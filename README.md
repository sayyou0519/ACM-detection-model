# ACM-detection-model
Random Forest-Based Hyperspectral Classification Pipeline

This repository contains the implementation code for performing pixel-wise classification of hyperspectral imagery using a pretrained Random Forest (RF) model.
The pipeline includes model inference, spectral band selection, RGB visualization, class overlay generation, and export of both GeoTIFF and high-resolution PNG outputs.

This code was developed for the study submitted to the IEEE Transactions on Geoscience and Remote Sensing (TGRS).

# Features

- Pixel-wise Random Forest classification
- Band selection for feature extraction
- RGB composite generation using selected spectral bands
- Prediction overlay mask generation
- Grayscale prediction visualization
- High-resolution PNG export
- GeoTIFF export preserving original metadata

  # Repository Structure

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



# Requirements
Install necessary Python packages:
```
pip install numpy rasterio joblib matplotlib
```

# How to Run
1. Prepare Inputs

Hyperspectral image (GeoTIFF)

Trained RF model (joblib format)

2. Execute the script
```  
python rf_predict.py
```

This will perform:

- Image loading
- Band selection
- RF model inference
- RGB visualization
- Class overlay creation
- GeoTIFF export
- 300 dpi PNG visualization export



  # Output files

 | File                     | Description                     |
| ------------------------ | ------------------------------- |
| `RF_prediction.tif`      | GeoTIFF predicted label map     |
| `RF_overlay_300dpi.png`  | RGB composite + Class 1 overlay |
| `RF_prediction_gray.png` | Grayscale label map             |
| `RF_class1_mask.png`     | Custom-colored mask for Class 1 |


