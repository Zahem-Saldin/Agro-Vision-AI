# Paddy_Predict_NewImage_Fixed.py
import os
import numpy as np
from tensorflow.keras.models import load_model
from skimage.transform import resize
import rasterio

epsilon = 1e-10
TARGET_SIZE = (128, 128)  # Must match CNN training input size
CNN_MODEL_PATH = "paddy_patch_based_softlabel_cnn_improved.keras"

# -------------------- INDEX CALCULATION --------------------
def compute_indices(bands):
    """
    Compute 6 indices from available bands
    bands: dict with band arrays {'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B11', 'B12'}
    Returns: list of 6 indices [NDVI, EVI, SAVI, NDWI, NDMI, GNDVI]
    """
    B2 = bands['B2'].astype(np.float32)
    B3 = bands['B3'].astype(np.float32)
    B4 = bands['B4'].astype(np.float32)
    B5 = bands['B5'].astype(np.float32)
    B6 = bands['B6'].astype(np.float32)
    B7 = bands['B7'].astype(np.float32)
    B8 = bands['B8'].astype(np.float32)
    B11 = bands['B11'].astype(np.float32)
    B12 = bands['B12'].astype(np.float32)

    ndvi  = (B8 - B4) / (B8 + B4 + epsilon)
    evi   = 2.5 * (B8 - B4) / (B8 + 6*B4 - 7.5*B2 + 1 + epsilon)
    savi  = ((B8 - B4) * 1.5) / (B8 + B4 + 1.5 + epsilon)
    ndwi  = (B8 - B11) / (B8 + B11 + epsilon)
    ndmi  = (B8 - B11) / (B8 + B11 + epsilon)
    gndvi = (B8 - B3) / (B8 + B3 + epsilon)

    indices = [ndvi, evi, savi, ndwi, ndmi, gndvi]
    indices = [np.nan_to_num(ind) for ind in indices]
    return indices


# -------------------- LOAD IMAGE --------------------
def load_image(image_path):
    with rasterio.open(image_path) as src:
        bands = {}
        # Only read bands actually available in the image
        band_names = ['B1','B2','B3','B4','B5','B6','B7','B8','B11','B12']
        for i, name in enumerate(band_names, start=1):
            if i <= src.count:
                bands[name] = src.read(i).astype(np.float32)

    indices = compute_indices(bands)

    # Stack indices into H x W x 6
    img_stack = np.stack(indices, axis=-1)

    # Normalize per band
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


# -------------------- PREDICT SOFT LABEL --------------------
def predict_growth_phase(image_path):
    img = load_image(image_path)
    img_batch = np.expand_dims(img, axis=0)
    prediction = model.predict(img_batch)

    # Flatten output if necessary
    prediction = np.ravel(prediction)
    if prediction.size != 3:
        # e.g., average over spatial patches
        prediction = np.mean(prediction.reshape(-1, 3), axis=0)

    phase_names = ['Sowing', 'Vegetative', 'Harvest']
    dominant_phase = phase_names[np.argmax(prediction)]
    return {"soft_labels": prediction, "dominant_phase": dominant_phase}



# -------------------- EXAMPLE USAGE --------------------
if __name__ == "__main__":
    new_image_path = r"C:/Users/Zahem Saldin/Desktop/GEE_Export_Paddy_Ad/S2_Polonnaruwa__BestImage_2022_5_P1.tif"
    result = predict_growth_phase(new_image_path)

    print("\n✅ Soft growth phase probabilities [p_sowing, p_vegetative, p_harvest]:")
    print(result["soft_labels"])
    print(f"🎯 Dominant growth phase: {result['dominant_phase']}")
