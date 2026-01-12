import os
import rasterio
import numpy as np

# -------------------- SETTINGS --------------------
input_folder = r"C:\Users\Zahem Saldin\Desktop\GEE_Export_Coconut_Ad"
output_folder_features = r"C:\Users\Zahem Saldin\Desktop\Coconut_CNN_Input"
output_folder_labels = r"C:\Users\Zahem Saldin\Desktop\Coconut_SoftLabels"

os.makedirs(output_folder_features, exist_ok=True)
os.makedirs(output_folder_labels, exist_ok=True)

epsilon = 1e-10


# -------------------- SOFT LABEL FUNCTION (COCONUT DENSITY) --------------------
def ndvi_to_soft_labels_coconut(ndvi):
    """
    Generate per-pixel soft labels [p_low, p_medium, p_high] for coconut density
    using NDVI as a proxy for canopy density.
    """
    ndvi_min = np.nanmin(ndvi)
    ndvi_max = np.nanmax(ndvi)
    ndvi_norm = (ndvi - ndvi_min) / (ndvi_max - ndvi_min + epsilon)

    # Low density (sparse canopy)
    p_low = np.clip(1 - 2 * ndvi_norm, 0, 1)

    # High density (dense canopy)
    p_high = np.clip(2 * ndvi_norm - 1, 0, 1)

    # Medium density (transition)
    p_medium = 1 - p_low - p_high

    # Normalize
    total = p_low + p_medium + p_high + epsilon
    p_low /= total
    p_medium /= total
    p_high /= total

    return np.stack([p_low, p_medium, p_high], axis=-1)


# -------------------- INDEX CALCULATION (COCONUT) --------------------
def compute_indices_coconut(bands):
    B2 = bands['B2'].astype(np.float32)   # Blue
    B3 = bands['B3'].astype(np.float32)   # Green
    B4 = bands['B4'].astype(np.float32)   # Red
    B8 = bands['B8'].astype(np.float32)   # NIR
    B11 = bands['B11'].astype(np.float32) # SWIR

    # Coconut-relevant indices
    ndvi = (B8 - B4) / (B8 + B4 + epsilon)
    evi = 2.5 * (B8 - B4) / (B8 + 6*B4 - 7.5*B2 + 1 + epsilon)
    gndvi = (B8 - B3) / (B8 + B3 + epsilon)
    savi = ((B8 - B4) * 1.5) / (B8 + B4 + 1.5 + epsilon)
    ndmi = (B8 - B11) / (B8 + B11 + epsilon)
    msi = B11 / (B8 + epsilon)

    indices = [ndvi, evi, gndvi, savi, ndmi, msi]
    indices = [np.nan_to_num(ind) for ind in indices]
    return indices


# -------------------- IMAGE PROCESSING --------------------
def process_image(image_path):
    with rasterio.open(image_path) as src:
        print(f"Processing: {os.path.basename(image_path)}")
        if src.count < 10:
            print(f"⚠ Skipping {image_path} - found {src.count} bands, need 10")
            return None

        # Read Sentinel-2 bands
        bands = {}
        for i, name in enumerate(['B1','B2','B3','B4','B5','B6','B7','B8','B11','B12'], start=1):
            bands[name] = src.read(i).astype(np.float32)

        # Compute coconut indices
        indices = compute_indices_coconut(bands)

        # NDVI for density-based soft labels
        ndvi = indices[0]
        soft_labels = ndvi_to_soft_labels_coconut(ndvi)

        return indices, soft_labels, src.meta


# -------------------- EXPORT TIFF --------------------
def export_tiff(output_path, bands_list, meta):
    new_meta = meta.copy()
    new_meta.update({'count': len(bands_list), 'dtype': rasterio.float32})
    with rasterio.open(output_path, 'w', **new_meta) as dst:
        for i, band in enumerate(bands_list, start=1):
            dst.write(band.astype(np.float32), i)


# -------------------- MAIN LOOP --------------------
tiff_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.tif')]

for tiff in tiff_files:
    image_path = os.path.join(input_folder, tiff)
    result = process_image(image_path)
    if result is None:
        continue

    indices, soft_labels, meta = result

    # Save CNN input features (6-band)
    out_features = os.path.join(output_folder_features, f"{os.path.splitext(tiff)[0]}_coconut_features.tif")
    export_tiff(out_features, indices, meta)

    # Save soft labels (3-band: low, medium, high density)
    out_labels = os.path.join(output_folder_labels, f"{os.path.splitext(tiff)[0]}_coconut_softlabels.tif")
    export_tiff(out_labels, [soft_labels[:, :, i] for i in range(3)], meta)

    print(f"✅ Saved features: {out_features}")
    print(f"✅ Saved coconut soft labels: {out_labels}")
