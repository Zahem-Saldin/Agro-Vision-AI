# Coconut_Predict_NewImage.py
import os
import numpy as np
from tensorflow.keras.models import load_model
from skimage.transform import resize
import rasterio

epsilon = 1e-10
TARGET_SIZE = (128, 128)  # Must match CNN training input size
CNN_MODEL_PATH = "coconut_patch_based_softlabel_cnn_improved.keras"

# -------------------- INDEX CALCULATION (Coconut) --------------------
def compute_indices_coconut(bands):
    """
    Compute 6 indices from available bands for coconut density estimation.
    Returns: list of 6 indices [NDVI, EVI, GNDVI, SAVI, NDMI, MSI]
    """
    B2 = bands['B2'].astype(np.float32)   # Blue
    B3 = bands['B3'].astype(np.float32)   # Green
    B4 = bands['B4'].astype(np.float32)   # Red
    B8 = bands['B8'].astype(np.float32)   # NIR
    B11 = bands['B11'].astype(np.float32) # SWIR

    ndvi  = (B8 - B4) / (B8 + B4 + epsilon)
    evi   = 2.5 * (B8 - B4) / (B8 + 6*B4 - 7.5*B2 + 1 + epsilon)
    gndvi = (B8 - B3) / (B8 + B3 + epsilon)
    savi  = ((B8 - B4) * 1.5) / (B8 + B4 + 1.5 + epsilon)
    ndmi  = (B8 - B11) / (B8 + B11 + epsilon)
    msi   = B11 / (B8 + epsilon)

    indices = [ndvi, evi, gndvi, savi, ndmi, msi]
    indices = [np.nan_to_num(ind) for ind in indices]
    return indices


# -------------------- LOAD IMAGE --------------------
def load_image(image_path):
    with rasterio.open(image_path) as src:
        bands = {}
        # Sentinel-2 bands needed for coconut indices
        band_names = ['B1','B2','B3','B4','B5','B6','B7','B8','B11','B12']
        for i, name in enumerate(band_names, start=1):
            if i <= src.count:
                bands[name] = src.read(i).astype(np.float32)

    indices = compute_indices_coconut(bands)

    # Stack indices into H x W x 6
    img_stack = np.stack(indices, axis=-1)

    # Normalize each band separately
    for i in range(img_stack.shape[-1]):
        band_min, band_max = np.min(img_stack[..., i]), np.max(img_stack[..., i])
        if band_max > band_min:
            img_stack[..., i] = (img_stack[..., i] - band_min) / (band_max - band_min + epsilon)
        else:
            img_stack[..., i] = 0.0

    # Resize to CNN input
    img_resized = resize(img_stack, (TARGET_SIZE[0], TARGET_SIZE[1], img_stack.shape[-1]),
                         mode='reflect', anti_aliasing=True)
    return img_resized.astype(np.float32)


# -------------------- LOAD CNN --------------------
model = load_model(CNN_MODEL_PATH)


# -------------------- PREDICT COCONUT DENSITY --------------------
def predict_coconut_density(image_path):
    img = load_image(image_path)
    img_batch = np.expand_dims(img, axis=0)
    prediction = model.predict(img_batch)

    # Flatten output if necessary
    prediction = np.ravel(prediction)
    if prediction.size != 3:
        # e.g., average over spatial patches
        prediction = np.mean(prediction.reshape(-1, 3), axis=0)

    density_classes = ['Low Density', 'Medium Density', 'High Density']
    dominant_density = density_classes[np.argmax(prediction)]
    return {"soft_labels": prediction, "dominant_density": dominant_density}


# -------------------- EXAMPLE USAGE --------------------
if __name__ == "__main__":
    new_image_path = r"C:/Users/Zahem Saldin/Desktop/GEE_Export_Coconut_Ad/S2_Coconut_106_BestImage_2019_3_P1.tif"
    result = predict_coconut_density(new_image_path)

    print("\n✅ Coconut canopy density probabilities [p_low, p_medium, p_high]:")
    print(result["soft_labels"])
    print(f"🌴 Dominant coconut density: {result['dominant_density']}")
