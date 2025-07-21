import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd

# --- Configuration ---
PATCH_SIZE = (256, 256)
STRIDE = 256
NUM_CLASSES = 4
IMG_CHANNELS = 2
EPOCHS = 10
BATCH_SIZE = 4

DATA_DIR = 'data'
TRAIN_EDGE = os.path.join(DATA_DIR, 'train/edge')
TRAIN_GABOR = os.path.join(DATA_DIR, 'train/gabor')
TRAIN_LABEL = os.path.join(DATA_DIR, 'train/label')
TEST_EDGE = os.path.join(DATA_DIR, 'test/edge')
TEST_GABOR = os.path.join(DATA_DIR, 'test/gabor')
TEST_LABEL = os.path.join(DATA_DIR, 'test/label')
RESULT_DIR = 'results/predictions'
os.makedirs(RESULT_DIR, exist_ok=True)

# --- Utility Functions ---

def load_image(path, grayscale=True):
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    return cv2.imread(path, flag)

def extract_patches(img, patch_size, stride):
    h, w = img.shape[:2]
    patches, positions = [], []
    for y in range(0, h - patch_size[1] + 1, stride):
        for x in range(0, w - patch_size[0] + 1, stride):
            patch = img[y:y + patch_size[1], x:x + patch_size[0]]
            patches.append(patch)
            positions.append((x, y))
    return patches, positions

def preprocess_label(label):
    return tf.one_hot(label, depth=NUM_CLASSES)

def load_patched_dataset(edge_dir, gabor_dir, label_dir):
    X, Y = [], []
    filenames = os.listdir(edge_dir)
    for fname in filenames:
        edge = load_image(os.path.join(edge_dir, fname))
        gabor = load_image(os.path.join(gabor_dir, fname))
        label = load_image(os.path.join(label_dir, fname))
        if edge is None or gabor is None or label is None:
            continue

        edge_patches, _ = extract_patches(edge, PATCH_SIZE, STRIDE)
        gabor_patches, _ = extract_patches(gabor, PATCH_SIZE, STRIDE)
        label_patches, _ = extract_patches(label, PATCH_SIZE, STRIDE)

        for e, g, l in zip(edge_patches, gabor_patches, label_patches):
            input_patch = np.stack([e / 255.0, g / 255.0], axis=-1)
            label_patch = l.astype(np.int32)
            X.append(input_patch)
            Y.append(label_patch)

    X = np.array(X, dtype=np.float32)
    Y = preprocess_label(np.array(Y))
    return X, Y

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

    edge_patches, positions = extract_patches(edge_img, patch_size, stride)
    gabor_patches, _ = extract_patches(gabor_img, patch_size, stride)

    for (x, y), e, g in zip(positions, edge_patches, gabor_patches):
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

# --- Training ---
print("🔄 Loading training data...")
X_train, Y_train = load_patched_dataset(TRAIN_EDGE, TRAIN_GABOR, TRAIN_LABEL)
print(f"✅ Training patches loaded: {X_train.shape}")

model = get_unet()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("🚀 Training model...")
model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)

# --- Testing ---
print("\n🧪 Running inference and evaluation...")
metrics_list = []
for fname in sorted(os.listdir(TEST_EDGE)):
    edge_img_path = os.path.join(TEST_EDGE, fname)
    gabor_img_path = os.path.join(TEST_GABOR, fname)
    label_img_path = os.path.join(TEST_LABEL, fname)

    if not os.path.exists(label_img_path):
        print(f"⚠️ Skipping {fname}: missing label.")
        continue

    edge_img = load_image(edge_img_path)
    gabor_img = load_image(gabor_img_path)
    label_img = load_image(label_img_path).astype(np.uint8)

    pred_mask = predict_full_image(model, edge_img, gabor_img, PATCH_SIZE, STRIDE)

    acc, miou, class_ious = compute_metrics(label_img, pred_mask, NUM_CLASSES)
    print(f"[{fname}] Acc: {acc:.4f}, mIoU: {miou:.4f}, IoUs: {[round(i,3) for i in class_ious]}")

    # Save prediction
    cv2.imwrite(os.path.join(RESULT_DIR, fname), pred_mask)

    metrics_list.append({
        "filename": fname,
        "accuracy": acc,
        "mean_iou": miou,
        "iou_class_0": class_ious[0],
        "iou_class_1": class_ious[1],
        "iou_class_2": class_ious[2],
        "iou_class_3": class_ious[3],
    })

# Save metrics to CSV
metrics_df = pd.DataFrame(metrics_list)
metrics_df.to_csv("results/metrics.csv", index=False)
print("\n✅ Predictions saved in: results/predictions/")
print("📊 Metrics saved in: results/metrics.csv")
