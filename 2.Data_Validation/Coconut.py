import os
import rasterio
import numpy as np

# Define the folder containing exported images
folder_path = r"C:\Users\Zahem Saldin\Desktop\GEE_Export_Coconut_Ad"

# List all TIFF files in the folder
tiff_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]


# Function to check for NaN values in bands and show band data
def check_for_nan_and_show_data(image_path):
    with rasterio.open(image_path) as dataset:
        for i in range(1, dataset.count + 1):  # Loop through bands
            band = dataset.read(i)

            # Print the band data (you can limit the output if needed, e.g., first few rows)
            print(f"Band {i} data for {os.path.basename(image_path)}:")
            print(band)

            if np.any(np.isnan(band)):  # Check if there are any NaN values
                print(f"Image {os.path.basename(image_path)} has NaN values in band {i}.")
                return True  # Return True if NaN values are found
    return False  # Return False if no NaN values


# Process each TIFF file and remove if NaN values are found
for tiff in tiff_files:
    image_path = os.path.join(folder_path, tiff)

    # If NaN values are found in any band, skip the image and remove it
    if check_for_nan_and_show_data(image_path):
        os.remove(image_path)  # Remove the image with NaN values
        print(f"Removed {tiff} due to NaN values.")
    else:
        # If image has no NaN values, proceed with further processing here
        print(f"Image {tiff} is valid and can proceed with further processing.")
        # Add any processing or exporting code here as needed
