import os
import rasterio
import numpy as np

# Define the folder containing the classified images
output_folder = r"C:\Users\Zahem Saldin\Desktop\GEE_Export_Paddy_Combined"

# List all subfolders (classified folders) in the output folder
class_folders = [f for f in os.listdir(output_folder) if os.path.isdir(os.path.join(output_folder, f))]

# Function to read and print the band values of a TIFF file and calculate the mean
def read_bands_from_image(image_path):
    with rasterio.open(image_path) as dataset:
        print(f"Reading: {os.path.basename(image_path)}")

        # Read the bands of the image
        bands = [dataset.read(i + 1) for i in range(dataset.count)]

        # Print the values of each band and calculate mean
        for i, band in enumerate(bands):
            band_mean = np.nanmean(band)  # Calculate the mean of the band, ignoring NaNs
            print(f"Band {i + 1} values: {band.flatten()[:10]}...")  # Display first 10 pixel values for preview
            print(f"Band {i + 1} mean value: {band_mean}")

        # Optionally, return the bands if needed for further processing
        return bands

# Iterate through each subfolder (classified folder)
for class_folder in class_folders:
    class_folder_path = os.path.join(output_folder, class_folder)

    # List all TIFF files in the current classified folder
    tiff_files = [f for f in os.listdir(class_folder_path) if f.endswith('.tif')]

    # Process each TIFF file
    for tiff in tiff_files:
        image_path = os.path.join(class_folder_path, tiff)

        # Read the bands from the image and print the values along with the mean
        read_bands_from_image(image_path)
