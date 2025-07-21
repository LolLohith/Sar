import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix
import pandas as pd

# Config
PATCH_SIZE = (256, 256)
STRIDE = 256
NUM_CLASSES = 4
IMG_CHANNELS = 2
EPOCHS = 10
BATCH_SIZE = 4

DATA_DIR = 'data'
# Configuration (Use raw strings for Windows paths)
TRAIN_EDGE = r"C:\Users\YourName\YourProject\data\train\edge"
TRAIN_GABOR = r"C:\Users\YourName\YourProject\data\train\gabor"
TRAIN_LABEL = r"C:\Users\YourName\YourProject\data\train\label"

TEST_EDGE = r"C:\Users\YourName\YourProject\data\test\edge"
TEST_GABOR = r"C:\Users\YourName\YourProject\data\test\gabor"
TEST_LABEL = r"C:\Users\YourName\YourProject\data\test\label"

RESULT_DIR = r"C:\Users\YourName\YourProject\results\predictions"
COLOR_DIR = r"C:\Users\YourName\YourProject\results\predictions_color"

#RESULT_DIR = 'results/predictions'
#COLOR_DIR = 'results/predictions_color'
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(COLOR_DIR, exist_ok=True)

# Helper Functions

def load_image(path, grayscale=True):
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(path, flag)
    if img is None:
        raise ValueError(f"Couldn't load image: {path}")
    return img

def extract_patches(img, patch_size, stride):
    h, w = img.shape[:2]
    patches, positions = [], []
    for y in range(0, h - patch_size[1] + 1, stride):
        for x in range(0, w - patch_size[0] + 1, stride):
            patch = img[y:y+patch_size[1], x:x+patch_size[0]]
            patches.append(patch)
            positions.append((x, y))
    return patches, positions

def preprocess_label(label):
    return tf.one_hot(label, depth=NUM_CLASSES)

def load_patched_dataset(edge_dir, gabor_dir, label_dir):
    X, Y = [], []
    for fname in os.listdir(edge_dir):
        edge = load_image(os.path.join(edge_dir, fname))
        gabor = load_image(os.path.join(gabor_dir, fname))
        label = load_image(os.path.join(label_dir, fname))
        if edge.shape != gabor.shape or edge.shape != label.shape:
            print(f"Skipping {fname}: mismatched dimensions.")
            continue
        e_patches, _ = extract_patches(edge, PATCH_SIZE, STRIDE)
        g_patches, _ = extract_patches(gabor, PATCH_SIZE, STRIDE)
        l_patches, _ = extract_patches(label, PATCH_SIZE, STRIDE)
        for e, g, l in zip(e_patches, g_patches, l_patches):
            X.append(np.stack([e / 255.0, g / 255.0], axis=-1))
            Y.append(l.astype(np.uint8))
    return np.array(X), preprocess_label(np.array(Y))

def get_unet(input_shape=(PATCH_SIZE[0], PATCH_SIZE[1], IMG_CHANNELS), num_classes=NUM_CLASSES):
    inputs = layers.Input(shape=input_shape)
    c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D()(c2)

    c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(c3)

    u1 = layers.UpSampling2D()(c3)
    u1 = layers.Concatenate()([u1, c2])
    c4 = layers.Conv2D(64, 3, activation='relu', padding='same')(u1)
    c4 = layers.Conv2D(64, 3, activation='relu', padding='same')(c4)

    u2 = layers.UpSampling2D()(c4)
    u2 = layers.Concatenate()([u2, c1])
    c5 = layers.Conv2D(32, 3, activation='relu', padding='same')(u2)
    c5 = layers.Conv2D(32, 3, activation='relu', padding='same')(c5)

    outputs = layers.Conv2D(num_classes, 1, activation='softmax')(c5)
    return models.Model(inputs, outputs)

def predict_full_image(model, edge_img, gabor_img, patch_size, stride):
    h, w = edge_img.shape[:2]
    pred_mask = np.zeros((h, w), dtype=np.float32)
    count_mask = np.zeros((h, w), dtype=np.float32)

    e_patches, positions = extract_patches(edge_img, patch_size, stride)
    g_patches, _ = extract_patches(gabor_img, patch_size, stride)

    for (x, y), e, g in zip(positions, e_patches, g_patches):
        input_patch = np.stack([e / 255.0, g / 255.0], axis=-1)[np.newaxis, ...]
        pred = model.predict(input_patch, verbose=0)[0]
        pred_class = np.argmax(pred, axis=-1)
        pred_mask[y:y+patch_size[1], x:x+patch_size[0]] += pred_class
        count_mask[y:y+patch_size[1], x:x+patch_size[0]] += 1

    count_mask[count_mask == 0] = 1
    return (pred_mask / count_mask).astype(np.uint8)

def compute_metrics(y_true, y_pred, num_classes):
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=list(range(num_classes)))
    ious = []
    for i in range(num_classes):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        union = TP + FP + FN
        iou = TP / union if union > 0 else 0
        ious.append(iou)
    acc = (y_true_flat == y_pred_flat).sum() / len(y_true_flat)
    return acc, np.mean(ious), ious

def colorize_mask(mask):
    colormap = np.array([
        [0, 0, 0],         # 0: background
        [255, 255, 0],     # 1: barren
        [0, 0, 255],       # 2: urban
        [0, 255, 0],       # 3: vegetation
    ], dtype=np.uint8)
    return colormap[mask]

# Training
print("Loading training data...")
X_train, Y_train = load_patched_dataset(TRAIN_EDGE, TRAIN_GABOR, TRAIN_LABEL)
print(f"Loaded {len(X_train)} patches.")

print("Building model...")
model = get_unet()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Training...")
model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)

# Testing
print("Testing...")
metrics_list = []

for fname in sorted(os.listdir(TEST_EDGE)):
    try:
        edge = load_image(os.path.join(TEST_EDGE, fname))
        gabor = load_image(os.path.join(TEST_GABOR, fname))
        label = load_image(os.path.join(TEST_LABEL, fname)).astype(np.uint8)

        if edge.shape != gabor.shape or edge.shape != label.shape:
            print(f"Skipping {fname}: mismatched shapes.")
            continue

        pred = predict_full_image(model, edge, gabor, PATCH_SIZE, STRIDE)
        acc, miou, ious = compute_metrics(label, pred, NUM_CLASSES)

        cv2.imwrite(os.path.join(RESULT_DIR, fname), pred)
        color_pred = colorize_mask(pred)
        cv2.imwrite(os.path.join(COLOR_DIR, fname), color_pred)

        metrics_list.append({
            "filename": fname,
            "accuracy": acc,
            "mean_iou": miou,
            "iou_class_0": ious[0],
            "iou_class_1": ious[1],
            "iou_class_2": ious[2],
            "iou_class_3": ious[3],
        })
        print(f"[{fname}] Acc: {acc:.4f}, mIoU: {miou:.4f}")
    except Exception as e:
        print(f"Error with {fname}: {e}")

df = pd.DataFrame(metrics_list)
df.to_csv("results/metrics.csv", index=False)
print("Saved metrics to results/metrics.csv")
print("Saved predictions to results/predictions/")
print("Saved color masks to results/predictions_color/")
