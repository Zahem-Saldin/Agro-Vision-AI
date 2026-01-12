# Paddy_Predict.py
import numpy as np
from tensorflow.keras.models import load_model
from skimage.transform import resize
import rasterio

# Function to extract multi-band image data and resize to uniform dimensions
def load_image_data(image_path, target_size=(256, 256)):
    with rasterio.open(image_path) as src:
        # Read all bands as channels
        bands = [src.read(i + 1) for i in range(src.count)]
        # Stack the bands into a 3D array (height, width, num_bands)
        image = np.stack(bands, axis=-1)
        # Resize the image to the target size
        image_resized = resize(image, (target_size[0], target_size[1], image.shape[-1]), mode='reflect', anti_aliasing=True)
    return image_resized

# Load the trained model
model = load_model("cultivation_level_model_paddy.h5")

# Function to classify a new image
def classify_image(image_path):
    image_data = load_image_data(image_path)
    image_data = image_data.astype('float32') / 255.0  # Normalize
    image_data = np.expand_dims(image_data, axis=0)  # Add batch dimension
    prediction = model.predict(image_data)
    predicted_class = np.argmax(prediction)
    return ['High', 'Medium', 'Low'][predicted_class]

# Example: Classifying a new image
new_image_path = "C:/Users/Zahem Saldin/Desktop/GEE_Export_Labled_Tea_02/Low/S2_Kotagala_BestImage_2024_4_P1_Low.tif"
cultivation_class = classify_image(new_image_path)
print(f"The cultivation level of the image is: {cultivation_class}")
