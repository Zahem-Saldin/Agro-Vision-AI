# Paddy_Train.py
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import save_model
from skimage.transform import resize
import rasterio

# Folder paths
base_folder = "C:/Users/Zahem Saldin/Desktop/GEE_Export_Labled_Tea"
subfolders = ['High', 'Medium', 'Low']

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

# Prepare dataset
X = []
y = []

for label in subfolders:
    folder_path = os.path.join(base_folder, label)
    for filename in os.listdir(folder_path):
        if filename.endswith('.tif'):
            image_path = os.path.join(folder_path, filename)
            image_data = load_image_data(image_path)
            X.append(image_data)
            y.append(label)

X = np.array(X)
y = np.array(y)

# Encode labels (High=0, Medium=1, Low=2)
label_mapping = {'High': 0, 'Medium': 1, 'Low': 2}
y_encoded = np.array([label_mapping[label] for label in y])

# Normalize pixel values to [0, 1]
X = X.astype('float32') / 255.0

# One-hot encode labels for CNN
y_encoded = to_categorical(y_encoded, num_classes=3)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Build CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:]))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='softmax'))  # 3 classes: High, Medium, Low

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# Save the trained model
model.save("cultivation_level_model_tea.h5")
print("Model saved as 'cultivation_level_model_Tea.h5'")
