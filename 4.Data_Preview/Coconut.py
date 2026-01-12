import os
import rasterio
import numpy as np

# Define the folder containing the updated classified images (with only mean values)
updated_folder = r"C:\Users\Zahem Saldin\Desktop\GEE_Export_Labled_Coconut"

# List all subfolders (classified folders) in the updated folder
class_folders = [f for f in os.listdir(updated_folder) if os.path.isdir(os.path.join(updated_folder, f))]

# Function to read and verify that the bands contain only the mean value
def verify_bands_from_updated_image(image_path):
    with rasterio.open(image_path) as dataset:
        print(f"Reading: {os.path.basename(image_path)}")

        # Read the bands of the image
        bands = [dataset.read(i + 1) for i in range(dataset.count)]

        # Check if each band contains only the mean value
        for i, band in enumerate(bands):
            unique_values = np.unique(band)  # Get unique values in the band
            print(f"Band {i + 1} unique values: {unique_values}")  # This should only print one value per band (the mean)

            # Ensure there is only one unique value in each band (i.e., the mean value)
            if len(unique_values) == 1:
                print(f"Band {i + 1} is consistent with the mean value: {unique_values[0]}")
            else:
                print(f"Warning: Band {i + 1} contains multiple values, which is unexpected.")

# Iterate through each subfolder (classified folder)
for class_folder in class_folders:
    class_folder_path = os.path.join(updated_folder, class_folder)

    # List all TIFF files in the current classified folder
    tiff_files = [f for f in os.listdir(class_folder_path) if f.endswith('.tif')]

    # Process each TIFF file
    for tiff in tiff_files:
        image_path = os.path.join(class_folder_path, tiff)

        # Verify the bands from the updated image (should only contain the mean value)
        verify_bands_from_updated_image(image_path)
