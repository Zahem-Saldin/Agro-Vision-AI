# Coconut_Train_Eval_Full.py
import os
import numpy as np
import rasterio
from skimage.util import view_as_windows
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Input, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# -------------------- SETTINGS --------------------
input_folder_features = r"C:/Users/Zahem Saldin/Desktop/Coconut_CNN_Input"
input_folder_labels   = r"C:/Users/Zahem Saldin/Desktop/Coconut_SoftLabels"

patch_size = 64
stride = 32
batch_size = 32
epochs = 50
learning_rate = 1e-3
min_class_fraction = 0.05  # min fraction of any class per patch

# -------------------- DATA LOADING --------------------
def load_tiff_as_array(path):
    with rasterio.open(path) as src:
        arr = np.stack([src.read(i+1) for i in range(src.count)], axis=-1)
    return arr.astype('float32')

def extract_balanced_patches(X_img, Y_img, patch_size=64, stride=32, min_class_fraction=0.05):
    H, W, C = X_img.shape
    patches_X = view_as_windows(X_img, (patch_size, patch_size, C), step=stride)
    patches_Y = view_as_windows(Y_img, (patch_size, patch_size, 3), step=stride)

    num_H, num_W = patches_X.shape[:2]
    patches_X = patches_X.reshape(num_H*num_W, patch_size, patch_size, C)
    patches_Y = patches_Y.reshape(num_H*num_W, patch_size, patch_size, 3)

    selected_X, selected_Y = [], []
    for px, py in zip(patches_X, patches_Y):
        fractions = py.mean(axis=(0,1))
        if (fractions > min_class_fraction).any():
            selected_X.append(px)
            selected_Y.append(py)
    return np.array(selected_X), np.array(selected_Y)

def standardize_per_image(img):
    for b in range(img.shape[-1]):
        mn, std = img[..., b].mean(), img[..., b].std()
        img[..., b] = (img[..., b] - mn) / (std + 1e-10)
    return img

def load_dataset(file_list):
    X_list, Y_list = [], []
    for f in file_list:
        feature_path = os.path.join(input_folder_features, f)
        label_path = os.path.join(input_folder_labels, f.replace('_features', '_softlabels'))
        if not os.path.exists(label_path):
            continue
        X_img = standardize_per_image(load_tiff_as_array(feature_path))
        Y_img = load_tiff_as_array(label_path)
        X_p, Y_p = extract_balanced_patches(X_img, Y_img, patch_size, stride, min_class_fraction)
        X_list.append(X_p)
        Y_list.append(Y_p)
    return np.concatenate(X_list, axis=0), np.concatenate(Y_list, axis=0)

# -------------------- TRAIN/TEST SPLIT --------------------
files = sorted([f for f in os.listdir(input_folder_features) if f.endswith('.tif')])
train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)

X_train, Y_train = load_dataset(train_files)
X_test,  Y_test  = load_dataset(test_files)

print(f"Train patches: {X_train.shape[0]}, Test patches: {X_test.shape[0]}")
print("Patch-level class distribution (sum of soft labels):", Y_train.reshape(-1,3).sum(axis=0))

# -------------------- TF DATA AUGMENTATION --------------------
def augment(X, Y):
    X = tf.cast(X, tf.float32)
    Y = tf.cast(Y, tf.float32)
    if tf.random.uniform(()) > 0.5:
        X = tf.image.flip_left_right(X)
        Y = tf.image.flip_left_right(Y)
    if tf.random.uniform(()) > 0.5:
        X = tf.image.flip_up_down(X)
        Y = tf.image.flip_up_down(Y)
    k = tf.random.uniform([], 0, 4, tf.int32)
    X = tf.image.rot90(X, k)
    Y = tf.image.rot90(Y, k)
    X = tf.image.random_brightness(X, 0.2)
    X = tf.image.random_contrast(X, 0.8, 1.2)
    return X, Y

train_ds = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
train_ds = train_ds.shuffle(5000).map(augment, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# -------------------- CNN MODEL --------------------
input_shape = X_train.shape[1:]
model = Sequential([
    Input(shape=input_shape),
    Conv2D(32, 3, activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(32, 3, activation='relu', padding='same'),
    BatchNormalization(),

    Conv2D(64, 3, activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, 3, activation='relu', padding='same'),
    BatchNormalization(),
    Dropout(0.4),

    Conv2D(128, 3, activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, 3, activation='relu', padding='same'),
    BatchNormalization(),
    Dropout(0.4),

    Conv2D(3, 1, activation='softmax', padding='same')
])

model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss=tf.keras.losses.KLDivergence(),
    metrics=['accuracy', tf.keras.metrics.KLDivergence(name="kl"), tf.keras.metrics.CategoricalCrossentropy(name="ce")]
)
model.summary()

# -------------------- CALLBACKS --------------------
csv_logger = CSVLogger("coconut_training_log.csv", append=False)
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss', verbose=1),
    ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-6, monitor='val_loss', verbose=1),
    ModelCheckpoint("best_coconut_model.keras", save_best_only=True, monitor='val_loss'),
    csv_logger
]

# -------------------- TRAIN --------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=callbacks
)

model.save("coconut_patch_based_softlabel_cnn_improved_01.keras")
print("✅ Final model saved")

# -------------------- PLOT METRICS --------------------
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend(); plt.title("Loss")
plt.subplot(1,3,2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend(); plt.title("Accuracy")
plt.subplot(1,3,3)
plt.plot(history.history['kl'], label='Train KL')
plt.plot(history.history['val_kl'], label='Val KL')
plt.legend(); plt.title("KL Divergence")
plt.tight_layout()
plt.show()

# -------------------- EVALUATION --------------------
def load_true_label(label_path):
    with rasterio.open(label_path) as src:
        soft_label = src.read().astype(np.float32)
        soft_label = np.transpose(soft_label, (1,2,0))
        avg_label = np.mean(soft_label.reshape(-1,3), axis=0)
        return np.argmax(avg_label)

def predict_image(image_path):
    img = load_tiff_as_array(image_path)
    img = standardize_per_image(img)
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img, verbose=0)
    pred = np.mean(pred.reshape(-1,3), axis=0)
    return np.argmax(pred), pred

def evaluate_model(features_folder, labels_folder, max_samples=None):
    y_true, y_pred = [], []
    feature_files = [f for f in os.listdir(features_folder) if f.endswith("_features.tif")]
    if max_samples:
        feature_files = feature_files[:max_samples]
    for f in feature_files:
        label_path = os.path.join(labels_folder, f.replace('_features','_softlabels'))
        if not os.path.exists(label_path):
            continue
        pred_class, _ = predict_image(os.path.join(features_folder, f))
        true_class = load_true_label(label_path)
        y_true.append(true_class)
        y_pred.append(pred_class)
        print(f"{f} | True={true_class}, Pred={pred_class}")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    labels_present = np.unique(np.concatenate([y_true, y_pred]))
    class_names = ["Low", "Medium", "High"]
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=labels_present, target_names=[class_names[i] for i in labels_present]))

    cm = confusion_matrix(y_true, y_pred, labels=labels_present)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[class_names[i] for i in labels_present],
                yticklabels=[class_names[i] for i in labels_present])
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
    plt.show()

# Run evaluation
evaluate_model(input_folder_features, input_folder_labels, max_samples=200)
