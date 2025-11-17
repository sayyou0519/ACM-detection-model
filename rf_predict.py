"""
Random Forest Prediction and Visualization Script
-------------------------------------------------
This script loads a hyperspectral image, applies a pretrained Random Forest (RF)
classifier to all pixels, and visualizes the resulting predictions.

Outputs:
1. GeoTIFF with predicted labels
2. RGB overlay (class mask on RGB bands)
3. Grayscale prediction PNG
4. Custom color mask PNG for Class 1

Author: Hyeji Sim
"""

import rasterio
import numpy as np
import joblib
import matplotlib.pyplot as plt
import time
import os


# =========================================================
# 1. File Paths
# =========================================================
img_path = r'G:\indoor\indoor_bamlight_prj.tif'    # Input hyperspectral image
model_path = r'G:\indoor\250720\RF_model.joblib'   # Trained Random Forest model


# =========================================================
# 2. Load Image & Metadata
# =========================================================
with rasterio.open(img_path) as src:
    img_data = src.read()            # Shape: (bands, rows, cols)
    profile = src.profile.copy()
    rows, cols = src.height, src.width

print(f"[INFO] Loaded image with shape: {img_data.shape}")


# =========================================================
# 3. Select Spectral Bands (Feature Engineering)
# =========================================================
# Example: use bands 10–277 (adjust as needed)
selected_band_indices = list(range(10, 278))
img_selected = img_data[selected_band_indices, :, :]

# Reshape for model input
img_2d = img_selected.reshape(img_selected.shape[0], -1).T   # (pixels, bands)
print(f"[INFO] Input for model: {img_2d.shape}")


# =========================================================
# 4. Load Model & Predict
# =========================================================
print("[INFO] Loading Random Forest model...")
rf_model = joblib.load(model_path)

print("[INFO] Running prediction on full image...")
start = time.time()
pred_flat = rf_model.predict(img_2d)
end = time.time()

pred_img = pred_flat.reshape(rows, cols)

# Timing summary
ms_per_img = (end - start) * 1000
mpix = (rows * cols) / 1_000_000
ms_per_mpix = ms_per_img / mpix
print(f"[INFO] Total time: {end - start:.2f} sec")
print(f"[INFO] ms/img: {ms_per_img:.2f} ms")
print(f"[INFO] ms/MPix: {ms_per_mpix:.2f} ms/MPix")


# =========================================================
# 5. Visualization (RGB + Class Overlay)
# =========================================================
# Example RGB composite (choose appropriate bands)
rgb_indices = [49, 29, 9]
rgb_raw = img_selected[rgb_indices].astype(np.float32)

# Normalize for display
rgb_norm = (rgb_raw - rgb_raw.min()) / (rgb_raw.max() - rgb_raw.min())
rgb_norm = np.transpose(rgb_norm, (1, 2, 0))  # (H, W, 3)

# Create overlay (Class 1 in Red)
class_mask = (pred_img == 1)
overlay = rgb_norm.copy()
overlay[class_mask] = [1, 0, 0]

alpha = 0.5
blended = (1 - alpha) * rgb_norm + alpha * overlay


# Display
plt.figure(figsize=(10, 8))
plt.imshow(blended)
plt.title("Random Forest Prediction Overlay (Class 1 in Red)")
plt.axis("off")
plt.show()


# Save PNG (high resolution)
fig, ax = plt.subplots(figsize=(12, 10))
ax.imshow(blended)
ax.axis("off")
plt.tight_layout()
overlay_png_path = "RF_overlay_300dpi.png"
fig.savefig(overlay_png_path, dpi=300, bbox_inches='tight', pad_inches=0)
plt.close(fig)
print(f"[INFO] Saved overlay PNG: {overlay_png_path}")


# =========================================================
# 6. Save Prediction as GeoTIFF
# =========================================================
profile.update({
    "count": 1,
    "dtype": pred_img.dtype,
    "compress": "lzw"
})

tif_out = "RF_prediction.tif"
with rasterio.open(tif_out, "w", **profile) as dst:
    dst.write(pred_img, 1)

print(f"[INFO] Saved GeoTIFF: {tif_out}")


# =========================================================
# 7. Save Grayscale Prediction
# =========================================================
plt.figure(figsize=(10, 8))
plt.imshow(pred_img, cmap="gray")
plt.axis("off")
gray_png = "RF_prediction_gray.png"
plt.savefig(gray_png, dpi=300, bbox_inches="tight", pad_inches=0)
plt.close()
print(f"[INFO] Saved grayscale PNG: {gray_png}")


# =========================================================
# 8. Save Class 1 Mask in Custom Color
# =========================================================
color_img = np.ones((rows, cols, 3), dtype=np.uint8) * 255
class_color = [211, 218, 245]  # Soft tone
color_img[class_mask] = class_color

plt.figure(figsize=(10, 8))
plt.imshow(color_img)
plt.axis("off")
mask_png = "RF_class1_mask.png"
plt.savefig(mask_png, dpi=300, bbox_inches="tight", pad_inches=0)
plt.close()
print(f"[INFO] Saved class mask PNG: {mask_png}")
