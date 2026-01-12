import os
import rasterio
import numpy as np

# -------------------- SETTINGS --------------------
input_folder = r"C:\Users\Zahem Saldin\Desktop\GEE_Export_Tea_Ad"
output_folder_features = r"C:\Users\Zahem Saldin\Desktop\Tea_CNN_Input"
output_folder_labels = r"C:\Users\Zahem Saldin\Desktop\Tea_SoftLabels"

os.makedirs(output_folder_features, exist_ok=True)
os.makedirs(output_folder_labels, exist_ok=True)

epsilon = 1e-10


# -------------------- SOFT LABEL FUNCTION (TEA) --------------------
def ndvi_to_soft_labels_tea(ndvi):
    """
    Generate per-pixel soft labels [p_plucked, p_flush, p_mature] for tea
    using NDVI as a proxy for canopy flush cycles.
    """
    ndvi_min = np.nanmin(ndvi)
    ndvi_max = np.nanmax(ndvi)
    ndvi_norm = (ndvi - ndvi_min) / (ndvi_max - ndvi_min + epsilon)

    # Stage 1: Recently plucked (low NDVI)
    p_plucked = np.clip(1 - 2 * ndvi_norm, 0, 1)

    # Stage 3: Mature canopy (high NDVI)
    p_mature = np.clip(2 * ndvi_norm - 1, 0, 1)

    # Stage 2: Flush (middle growth)
    p_flush = 1 - p_plucked - p_mature

    # Normalize
    total = p_plucked + p_flush + p_mature + epsilon
    p_plucked /= total
    p_flush /= total
    p_mature /= total

    return np.stack([p_plucked, p_flush, p_mature], axis=-1)


# -------------------- INDEX CALCULATION (TEA-TUNED) --------------------
def compute_indices_tea(bands):
    B2 = bands['B2'].astype(np.float32)   # Blue
    B3 = bands['B3'].astype(np.float32)   # Green
    B4 = bands['B4'].astype(np.float32)   # Red
    B5 = bands['B5'].astype(np.float32)   # Red Edge
    B8 = bands['B8'].astype(np.float32)   # NIR
    B11 = bands['B11'].astype(np.float32) # SWIR

    # Tea-relevant indices
    ndvi = (B8 - B4) / (B8 + B4 + epsilon)                 # Canopy vigor
    gndvi = (B8 - B3) / (B8 + B3 + epsilon)                # Chlorophyll
    evi = 2.5 * (B8 - B4) / (B8 + 6*B4 - 7.5*B2 + 1 + epsilon) # Dense canopy
    cigreen = (B8 / (B3 + epsilon)) - 1                    # Chlorophyll index green
    cire = (B8 / (B5 + epsilon)) - 1                       # Chlorophyll index red-edge
    msi = B11 / (B8 + epsilon)                             # Moisture stress index

    indices = [ndvi, gndvi, evi, cigreen, cire, msi]
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

        # Compute tea indices
        indices = compute_indices_tea(bands)

        # NDVI for tea soft labels
        ndvi = indices[0]
        soft_labels = ndvi_to_soft_labels_tea(ndvi)

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
    out_features = os.path.join(output_folder_features, f"{os.path.splitext(tiff)[0]}_tea_features.tif")
    export_tiff(out_features, indices, meta)

    # Save soft labels (3-band: plucked, flush, mature)
    out_labels = os.path.join(output_folder_labels, f"{os.path.splitext(tiff)[0]}_tea_softlabels.tif")
    export_tiff(out_labels, [soft_labels[:, :, i] for i in range(3)], meta)

    print(f"✅ Saved features: {out_features}")
    print(f"✅ Saved tea soft labels: {out_labels}")
