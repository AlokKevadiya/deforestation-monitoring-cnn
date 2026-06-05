#!/usr/bin/env python3
# Deforestation Monitoring Using Deep Learning - Alok Prakashbhai Kevadiya
# Standalone script version of deforestation_cnn.ipynb

# # Deforestation Monitoring Using Deep Learning
# ### A CNN-Based Forest / Non-Forest Classification of Aerial Imagery
#
# **Student:** Alok Prakashbhai Kevadiya  
# **Field:** Remote Sensing and Satellite Image Analysis  
# **Course:** Machine Learning — Phase 2  
# **Institution:** University of Europe for Applied Sciences
#
# ---
#
# **Dataset (instructor-approved):** [Forest Aerial Images for Segmentation](https://www.kaggle.com/datasets/quadeer15sh/augmented-forest-segmentation)
#
# The dataset was published for *semantic segmentation* (each image has a forest/non-forest mask).
# In this notebook we **adapt it into a binary image-classification dataset**: each image is labelled
# **Forest** or **Non-Forest** based on the fraction of forest pixels in its mask (threshold = 0.5).
#
# > **Note:** The instructor (Raja Hashim Ali) approved this dataset on Microsoft Teams. The approval
# > screenshot is included in the proposal document and the GitHub README, as required.
#
# This notebook covers the full required pipeline: data loading, preprocessing, resizing & normalization,
# train/val/test split, augmentation, a custom CNN, training, evaluation, confusion matrix, accuracy/loss
# curves, classification report, a transfer-learning comparison (MobileNetV2), ROC curves, a model
# comparison table, Grad-CAM, and saving of all figures.

# ## 1. Imports and Configuration

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, auc, ConfusionMatrixDisplay)
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Configuration
IMG_SIZE   = 128          # images resized to 128x128
BATCH_SIZE = 32
EPOCHS     = 25
FOREST_THRESHOLD = 0.5    # >=50% forest pixels in mask => "Forest"
CLASS_NAMES = ["Non-Forest", "Forest"]   # index 0, index 1

# Folder to save all figures (also committed to the GitHub repo)
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

# ## 2. Locate the Dataset
#
# On Kaggle, after you click **Add Input** and attach the dataset, it is mounted under
# `/kaggle/input/`. The cell below auto-detects the image and mask folders so the notebook is portable.
# The dataset contains an `images/` folder and a `masks/` folder with matching filenames.

# Auto-detect the dataset root under /kaggle/input
BASE = "/kaggle/input"
images_dir, masks_dir = None, None

for root, dirs, files in os.walk(BASE):
    low = os.path.basename(root).lower()
    if low in ("images", "image", "imgs") and images_dir is None:
        images_dir = root
    if low in ("masks", "mask", "annotations") and masks_dir is None:
        masks_dir = root

# Fallback: if the structure differs, set these two paths manually.
assert images_dir and masks_dir, (
    "Could not auto-detect images/masks folders. "
    "Set images_dir and masks_dir manually after inspecting /kaggle/input."
)

print("Images dir:", images_dir)
print("Masks  dir:", masks_dir)
print("Num image files:", len(os.listdir(images_dir)))
print("Num mask  files:", len(os.listdir(masks_dir)))

# ## 3. Generate Classification Labels (Segmentation -> Classification)
#
# For every image we load its mask, compute the fraction of forest pixels, and assign a single label:
#
# * fraction of forest pixels **>= 0.5**  -> **Forest** (label 1)
# * otherwise                              -> **Non-Forest** (label 0)
#
# This converts the segmentation dataset into the clean binary classification dataset the project needs.

def list_pairs(images_dir, masks_dir):
    """Match image files to mask files by stem (filename without extension)."""
    img_files  = {os.path.splitext(f)[0]: os.path.join(images_dir, f)
                  for f in os.listdir(images_dir)}
    mask_files = {os.path.splitext(f)[0]: os.path.join(masks_dir, f)
                  for f in os.listdir(masks_dir)}
    common = sorted(set(img_files) & set(mask_files))
    return [(img_files[k], mask_files[k]) for k in common]

pairs = list_pairs(images_dir, masks_dir)
print("Matched image/mask pairs:", len(pairs))

def mask_forest_fraction(mask_path):
    """Return fraction of 'forest' pixels in a mask (mask is white where forest)."""
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    forest = (m > 127).astype(np.uint8)   # white pixels = forest
    return forest.mean()

# Build a labels dataframe
records = []
for img_path, mask_path in pairs:
    frac = mask_forest_fraction(mask_path)
    if frac is None:
        continue
    label = 1 if frac >= FOREST_THRESHOLD else 0
    records.append({"image": img_path, "forest_fraction": frac, "label": label})

df = pd.DataFrame(records)
print("Usable samples:", len(df))
print(df["label"].value_counts().rename({0: "Non-Forest", 1: "Forest"}))
df.head()

# ## 4. Class Distribution

counts = df["label"].value_counts().sort_index()
plt.figure(figsize=(5, 4))
plt.bar([CLASS_NAMES[i] for i in counts.index], counts.values,
        color=["#c0392b", "#27ae60"])
plt.title("Class Distribution")
plt.ylabel("Number of images")
for i, v in zip(range(len(counts)), counts.values):
    plt.text(i, v + max(counts.values)*0.01, str(v), ha="center")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/01_class_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

# ## 5. Sample Images from Each Class

def load_image(path, size=IMG_SIZE):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    return img

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for row, lab in enumerate([1, 0]):                       # Forest first, then Non-Forest
    sample = df[df["label"] == lab].sample(4, random_state=SEED)
    for col, (_, r) in enumerate(sample.iterrows()):
        axes[row, col].imshow(load_image(r["image"]))
        axes[row, col].set_title(f"{CLASS_NAMES[lab]}\n(forest={r['forest_fraction']:.2f})", fontsize=9)
        axes[row, col].axis("off")
plt.suptitle("Sample Images by Class", fontsize=14)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/02_sample_images.png", dpi=150, bbox_inches="tight")
plt.show()

# ## 6. Load, Resize, and Normalize All Images
#
# Each image is resized to 128x128 and pixel values scaled to [0, 1].
# For a dataset of this size this fits comfortably in memory as a NumPy array.

X = np.zeros((len(df), IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
y = df["label"].values.astype(np.float32)

for i, path in enumerate(df["image"].values):
    X[i] = load_image(path) / 255.0      # normalize to [0, 1]
    if (i + 1) % 500 == 0:
        print(f"Loaded {i+1}/{len(df)} images")

print("X shape:", X.shape, "| y shape:", y.shape)

# ## 7. Train / Validation / Test Split
#
# Stratified split: ~70% train, ~15% validation, ~15% test, preserving class balance in each subset.

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=SEED)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED)

print("Train:", X_train.shape[0])
print("Val:  ", X_val.shape[0])
print("Test: ", X_test.shape[0])

# Class weights to counter any imbalance
from sklearn.utils.class_weight import compute_class_weight
cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
class_weight = {0: cw[0], 1: cw[1]}
print("Class weights:", class_weight)

# ## 8. Data Augmentation
#
# Augmentation is applied to the training set only, using a Keras preprocessing pipeline:
# random flips, rotation, zoom, and translation. Aerial imagery has no fixed orientation,
# so flips and rotations are natural and effective.

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomTranslation(0.1, 0.1),
], name="data_augmentation")

# Visualize augmentation on one image
sample_img = X_train[0:1]
plt.figure(figsize=(10, 4))
for i in range(8):
    aug = data_augmentation(sample_img, training=True)[0].numpy()
    plt.subplot(2, 4, i + 1)
    plt.imshow(np.clip(aug, 0, 1))
    plt.axis("off")
plt.suptitle("Data Augmentation Examples")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/03_augmentation_examples.png", dpi=150, bbox_inches="tight")
plt.show()

# ## 9. Custom CNN Model
#
# A sequential CNN with three convolutional blocks (Conv + BatchNorm + MaxPool), followed by
# global average pooling, a dense layer with dropout, and a sigmoid output for binary classification.

def build_custom_cnn():
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        data_augmentation,

        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid"),
    ], name="Custom_CNN")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

cnn = build_custom_cnn()
cnn.summary()

# ## 10. Train the Custom CNN

callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ModelCheckpoint("best_custom_cnn.keras", monitor="val_accuracy", save_best_only=True),
]

history = cnn.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1,
)

# ## 11. Accuracy and Loss Curves (Custom CNN)

def plot_history(history, title, fname):
    h = history.history
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(h["accuracy"], label="train")
    ax[0].plot(h["val_accuracy"], label="val")
    ax[0].set_title(f"{title} — Accuracy"); ax[0].set_xlabel("Epoch"); ax[0].legend()
    ax[1].plot(h["loss"], label="train")
    ax[1].plot(h["val_loss"], label="val")
    ax[1].set_title(f"{title} — Loss"); ax[1].set_xlabel("Epoch"); ax[1].legend()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{fname}", dpi=150, bbox_inches="tight")
    plt.show()

plot_history(history, "Custom CNN", "04_custom_cnn_curves.png")

# ## 12. Evaluation Helper (Confusion Matrix, Report, ROC)

def evaluate_model(model, X_test, y_test, name, idx):
    """Evaluate a model and save confusion matrix + ROC. Returns a metrics dict."""
    probs = model.predict(X_test).ravel()
    preds = (probs >= 0.5).astype(int)

    # --- Classification report ---
    print(f"\n===== {name} — Classification Report =====")
    print(classification_report(y_test, preds, target_names=CLASS_NAMES, digits=4))
    report = classification_report(y_test, preds, target_names=CLASS_NAMES,
                                   output_dict=True, digits=4)

    # --- Confusion matrix ---
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    disp.plot(cmap="Greens", values_format="d")
    plt.title(f"{name} — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{idx}a_{name.replace(' ', '_')}_confusion.png",
                dpi=150, bbox_inches="tight")
    plt.show()

    # --- ROC curve ---
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"{name} — ROC Curve"); plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{idx}b_{name.replace(' ', '_')}_roc.png",
                dpi=150, bbox_inches="tight")
    plt.show()

    acc = (preds == y_test).mean()
    return {
        "Model": name,
        "Accuracy": acc,
        "Precision": report["Forest"]["precision"],
        "Recall": report["Forest"]["recall"],
        "F1-score": report["Forest"]["f1-score"],
        "AUC": roc_auc,
        "probs": probs,
    }

cnn_metrics = evaluate_model(cnn, X_test, y_test, "Custom CNN", "05")

# ## 13. Transfer Learning Model — MobileNetV2
#
# A MobileNetV2 base pretrained on ImageNet (top removed, frozen) with a new binary head on top.
# Inputs are scaled to MobileNetV2's expected range with `preprocess_input`.

def build_transfer_model():
    base = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                       include_top=False, weights="imagenet")
    base.trainable = False    # freeze the pretrained feature extractor

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = data_augmentation(inputs)
    x = layers.Rescaling(255.0)(x)          # undo our [0,1] scaling -> [0,255]
    x = preprocess_input(x)                 # MobileNetV2 preprocessing -> [-1,1]
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs, name="MobileNetV2_Transfer")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model

tl = build_transfer_model()
tl.summary()

# ## 14. Train the Transfer-Learning Model

tl_callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ModelCheckpoint("best_transfer.keras", monitor="val_accuracy", save_best_only=True),
]

tl_history = tl.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight,
    callbacks=tl_callbacks,
    verbose=1,
)

plot_history(tl_history, "MobileNetV2 Transfer", "06_transfer_curves.png")

# ## 15. Evaluate the Transfer-Learning Model

tl_metrics = evaluate_model(tl, X_test, y_test, "MobileNetV2 Transfer", "07")

# ## 16. Model Comparison Table

comparison = pd.DataFrame([
    {k: v for k, v in cnn_metrics.items() if k != "probs"},
    {k: v for k, v in tl_metrics.items() if k != "probs"},
]).set_index("Model").round(4)

print(comparison)

# Save the comparison table as a figure
fig, ax = plt.subplots(figsize=(9, 2))
ax.axis("off")
tbl = ax.table(cellText=comparison.reset_index().values,
               colLabels=comparison.reset_index().columns,
               cellLoc="center", loc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 1.6)
plt.title("Model Comparison", pad=12)
plt.savefig(f"{FIG_DIR}/08_model_comparison_table.png", dpi=150, bbox_inches="tight")
plt.show()

# Combined ROC comparison
fpr1, tpr1, _ = roc_curve(y_test, cnn_metrics["probs"])
fpr2, tpr2, _ = roc_curve(y_test, tl_metrics["probs"])
plt.figure(figsize=(6, 5))
plt.plot(fpr1, tpr1, label=f"Custom CNN (AUC={auc(fpr1,tpr1):.3f})")
plt.plot(fpr2, tpr2, label=f"MobileNetV2 (AUC={auc(fpr2,tpr2):.3f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison"); plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/09_roc_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# ## 17. Grad-CAM Visualization
#
# Grad-CAM highlights the image regions that most influenced the prediction. We apply it to the
# custom CNN's last convolutional layer to confirm the model focuses on forest regions.

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        loss = preds[:, 0]
    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heatmap = conv_out @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

# Find the last conv layer
last_conv = [l.name for l in cnn.layers if isinstance(l, layers.Conv2D)][-1]
print("Last conv layer:", last_conv)

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
samples = np.random.choice(len(X_test), 4, replace=False)
for col, idx in enumerate(samples):
    img = X_test[idx:idx+1]
    heatmap = make_gradcam_heatmap(img, cnn, last_conv)
    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap_color = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) / 255.0
    overlay = np.clip(0.6*X_test[idx] + 0.4*heatmap_color, 0, 1)

    pred = cnn.predict(img, verbose=0)[0, 0]
    true = CLASS_NAMES[int(y_test[idx])]
    axes[0, col].imshow(X_test[idx]); axes[0, col].axis("off")
    axes[0, col].set_title(f"True: {true}", fontsize=10)
    axes[1, col].imshow(overlay); axes[1, col].axis("off")
    axes[1, col].set_title(f"Grad-CAM\nP(Forest)={pred:.2f}", fontsize=10)
plt.suptitle("Grad-CAM — Custom CNN", fontsize=14)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/10_gradcam.png", dpi=150, bbox_inches="tight")
plt.show()

# ## 18. Error Analysis
#
# We inspect a few misclassified test images to understand where the model struggles
# (typically images near the 50% forest threshold, where the class boundary is ambiguous).

preds = (tl_metrics["probs"] >= 0.5).astype(int)
mis = np.where(preds != y_test.astype(int))[0]
print(f"Misclassified (MobileNetV2): {len(mis)} / {len(y_test)}")

if len(mis) > 0:
    show = mis[:4]
    plt.figure(figsize=(14, 4))
    for i, idx in enumerate(show):
        plt.subplot(1, 4, i+1)
        plt.imshow(X_test[idx]); plt.axis("off")
        plt.title(f"True: {CLASS_NAMES[int(y_test[idx])]}\n"
                  f"Pred: {CLASS_NAMES[preds[idx]]}", fontsize=9)
    plt.suptitle("Misclassified Examples")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/11_error_analysis.png", dpi=150, bbox_inches="tight")
    plt.show()

# ## 19. Summary
#
# * The segmentation dataset was successfully adapted into a binary classification task.
# * A custom CNN and a MobileNetV2 transfer-learning model were trained and compared.
# * All required outputs were produced and saved to the `figures/` folder: class distribution,
#   sample images, augmentation examples, accuracy/loss curves, confusion matrices, classification
#   reports, ROC curves, a model comparison table, Grad-CAM, and error analysis.
# * See the proposal document for the full discussion of methodology, research questions, and expected results.
#
# **Instructor approval of the dataset is documented in the proposal and README.**

print("Saved figures:")
for f in sorted(os.listdir(FIG_DIR)):
    print(" -", f)
