import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
import rasterio
from sklearn.model_selection import train_test_split

# -------------------- SETTINGS --------------------
TEST_SAMPLE_COUNT = 100  # number of patches to test

# Load your saved model
model = tf.keras.models.load_model("coconut_patch_based_softlabel_cnn_improved_2.keras")

# Function to load and process test patches (reuse from training script)
def load_tiff_as_array(path):
    with rasterio.open(path) as src:
        arr = np.stack([src.read(i+1) for i in range(src.count)], axis=-1)
    return arr.astype('float32')

def standardize_per_image(img):
    for b in range(img.shape[-1]):
        mn, std = img[..., b].mean(), img[..., b].std()
        img[..., b] = (img[..., b] - mn) / (std + 1e-10)
    return img

def extract_patches(image, patch_size=64, stride=32):
    from skimage.util import view_as_windows
    H, W, C = image.shape
    patches = view_as_windows(image, (patch_size, patch_size, C), step=stride)
    num_H, num_W = patches.shape[:2]
    return patches.reshape(num_H*num_W, patch_size, patch_size, C)

# Load test dataset
input_folder_features = r"C:/Users/Zahem Saldin/Desktop/Research/Coconut_CNN_Input"
input_folder_labels = r"C:/Users/Zahem Saldin/Desktop/Research/Coconut_SoftLabels"

files = sorted([f for f in os.listdir(input_folder_features) if f.endswith('.tif')])
_, test_files = train_test_split(files, test_size=0.2, random_state=42)

# Gather test patches
X_test_list, Y_test_list = [], []
for f in test_files:
    feature_path = os.path.join(input_folder_features, f)
    label_path = os.path.join(input_folder_labels, f.replace('_features', '_softlabels'))
    if not os.path.exists(label_path):
        continue
    X_img = standardize_per_image(load_tiff_as_array(feature_path))
    Y_img = load_tiff_as_array(label_path)
    X_patches = extract_patches(X_img, patch_size=64, stride=32)
    Y_patches = extract_patches(Y_img, patch_size=64, stride=32)
    X_test_list.append(X_patches)
    Y_test_list.append(Y_patches)

X_test = np.concatenate(X_test_list, axis=0)[:TEST_SAMPLE_COUNT]
Y_test = np.concatenate(Y_test_list, axis=0)[:TEST_SAMPLE_COUNT]

# -------------------- PREDICTIONS --------------------
y_pred_soft = model.predict(X_test)
# Convert softmax outputs to class indices
y_pred = np.argmax(y_pred_soft, axis=-1).flatten()
y_true = np.argmax(Y_test, axis=-1).flatten()

# -------------------- CONFUSION MATRIX --------------------
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)

# Optional: classification report
print("\nClassification Report:\n", classification_report(y_true, y_pred))

# Optional: plot confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
