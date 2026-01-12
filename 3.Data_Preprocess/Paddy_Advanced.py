import os
import rasterio
import numpy as np


input_folder = r"C:\Users\Zahem Saldin\Desktop\GEE_Export_Paddy_Ad"
output_folder_features = r"C:\Users\Zahem Saldin\Desktop\GEE_Export_CNN_Input"
output_folder_labels = r"C:\Users\Zahem Saldin\Desktop\GEE_Export_SoftLabels"

os.makedirs(output_folder_features, exist_ok=True)
os.makedirs(output_folder_labels, exist_ok=True)

epsilon = 1e-10


def ndvi_to_soft_labels(ndvi):
    
    ndvi_min = np.nanmin(ndvi)
    ndvi_max = np.nanmax(ndvi)
    ndvi_norm = (ndvi - ndvi_min) / (ndvi_max - ndvi_min + epsilon)

    p_sowing = np.clip(1 - 2 * ndvi_norm, 0, 1)
    p_harvest = np.clip(2 * ndvi_norm - 1, 0, 1)
    p_vegetative = 1 - p_sowing - p_harvest

    total = p_sowing + p_vegetative + p_harvest + epsilon
    p_sowing /= total
    p_vegetative /= total
    p_harvest /= total

    soft_labels = np.stack([p_sowing, p_vegetative, p_harvest], axis=-1)
    return soft_labels

def compute_indices(bands):
    B2 = bands['B2'].astype(np.float32)
    B3 = bands['B3'].astype(np.float32)
    B4 = bands['B4'].astype(np.float32)
    B8 = bands['B8'].astype(np.float32)
    B11 = bands['B11'].astype(np.float32)

    ndvi = (B8 - B4) / (B8 + B4 + epsilon)
    evi = 2.5 * (B8 - B4) / (B8 + 6*B4 - 7.5*B2 + 1 + epsilon)
    savi = ((B8 - B4) * 1.5) / (B8 + B4 + 1.5 + epsilon)
    ndwi = (B8 - B11) / (B8 + B11 + epsilon)
    ndmi = (B8 - B11) / (B8 + B11 + epsilon)
    gndvi = (B8 - B3) / (B8 + B3 + epsilon)

    indices = [ndvi, evi, savi, ndwi, ndmi, gndvi]
    indices = [np.nan_to_num(ind) for ind in indices]
    return indices


def process_image(image_path):
    with rasterio.open(image_path) as src:
        print(f"Processing: {os.path.basename(image_path)}")
        num_bands = src.count
        if num_bands < 10:
            print(f"⚠ Skipping {image_path} - found {num_bands} bands, need 10")
            return None

       
        bands = {}
        for i, name in enumerate(['B1','B2','B3','B4','B5','B6','B7','B8','B11','B12'], start=1):
            bands[name] = src.read(i).astype(np.float32)

  
        indices = compute_indices(bands)

      
        ndvi = indices[0]
        soft_labels = ndvi_to_soft_labels(ndvi)

        return indices, soft_labels, src.meta


def export_tiff(output_path, bands_list, meta):
    new_meta = meta.copy()
    new_meta.update({'count': len(bands_list), 'dtype': rasterio.float32})
    with rasterio.open(output_path, 'w', **new_meta) as dst:
        for i, band in enumerate(bands_list, start=1):
            dst.write(band.astype(np.float32), i)


tiff_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.tif')]

for tiff in tiff_files:
    image_path = os.path.join(input_folder, tiff)
    result = process_image(image_path)
    if result is None:
        continue

    indices, soft_labels, meta = result


    out_features = os.path.join(output_folder_features, f"{os.path.splitext(tiff)[0]}_features.tif")
    export_tiff(out_features, indices, meta)

 
    out_labels = os.path.join(output_folder_labels, f"{os.path.splitext(tiff)[0]}_softlabels.tif")
    export_tiff(out_labels, [soft_labels[:, :, i] for i in range(3)], meta)

    print(f"Saved features: {out_features}")
    print(f"Saved soft labels: {out_labels}")
