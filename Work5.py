import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix
import pandas as pd

# --- CONFIG ---
PATCH_SIZE = (256, 256)
STRIDE = 128
NUM_CLASSES = 4
IMG_CHANNELS = 2
EPOCHS = 50
BATCH_SIZE = 4

# --- PATHS (Change accordingly) ---
BASE_PATH = r"C:\Users\YourName\YourProject"
TRAIN_EDGE = os.path.join(BASE_PATH, "data/train/edge")
TRAIN_GABOR = os.path.join(BASE_PATH, "data/train/gabor")
TRAIN_LABEL = os.path.join(BASE_PATH, "data/train/label")
TEST_EDGE = os.path.join(BASE_PATH, "data/test/edge")
TEST_GABOR = os.path.join(BASE_PATH, "data/test/gabor")
TEST_LABEL = os.path.join(BASE_PATH, "data/test/label")
RESULT_DIR = os.path.join(BASE_PATH, "results/predictions")
COLOR_DIR = os.path.join(BASE_PATH, "results/predictions_color")
COMPARE_DIR = os.path.join(BASE_PATH, "results/comparisons")
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(COLOR_DIR, exist_ok=True)
os.makedirs(COMPARE_DIR, exist_ok=True)

# --- HELPERS ---
def load_image(path, grayscale=True):
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(path, flag)
    if img is None:
        raise ValueError(f"Couldn't load image: {path}")
    return img

def extract_patches_with_padding(img, patch_size, stride):
    h, w = img.shape[:2]
    pad_h = (patch_size[1] - h % patch_size[1]) % patch_size[1]
    pad_w = (patch_size[0] - w % patch_size[0]) % patch_size[0]
    img_padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    patches, positions = [], []
    for y in range(0, img_padded.shape[0] - patch_size[1] + 1, stride):
        for x in range(0, img_padded.shape[1] - patch_size[0] + 1, stride):
            patches.append(img_padded[y:y+patch_size[1], x:x+patch_size[0]])
            positions.append((x, y))
    return patches, positions, img_padded.shape[:2]

def preprocess_label(label):
    label = label.copy()
    label[label == 0] = 255  # Mark unlabeled
    label = label - 1
    label[label > 3] = 255
    return label.astype(np.uint8)

def load_patched_dataset(edge_dir, gabor_dir, label_dir):
    X, Y = [], []
    for fname in os.listdir(edge_dir):
        edge = load_image(os.path.join(edge_dir, fname))
        gabor = load_image(os.path.join(gabor_dir, fname))
        label = preprocess_label(load_image(os.path.join(label_dir, fname)))

        if edge.shape != gabor.shape or edge.shape != label.shape:
            print(f"Skipping {fname}: mismatched dimensions.")
            continue

        e_patches, _, _ = extract_patches_with_padding(edge, PATCH_SIZE, STRIDE)
        g_patches, _, _ = extract_patches_with_padding(gabor, PATCH_SIZE, STRIDE)
        l_patches, _, _ = extract_patches_with_padding(label, PATCH_SIZE, STRIDE)

        for e, g, l in zip(e_patches, g_patches, l_patches):
            mask = l != 255
            onehot = tf.one_hot(np.clip(l, 0, NUM_CLASSES-1), depth=NUM_CLASSES).numpy()
            onehot[~mask] = 0  # ignore unlabeled
            X.append(np.stack([e / 255.0, g / 255.0], axis=-1))
            Y.append(onehot)
    return np.array(X), np.array(Y)

def masked_categorical_crossentropy(y_true, y_pred):
    mask = tf.reduce_sum(y_true, axis=-1) > 0
    y_true_masked = tf.boolean_mask(y_true, mask)
    y_pred_masked = tf.boolean_mask(y_pred, mask)
    return tf.keras.losses.categorical_crossentropy(y_true_masked, y_pred_masked)

def get_unet(input_shape=(256, 256, 2), num_classes=4):
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
    e_patches, positions, padded_shape = extract_patches_with_padding(edge_img, patch_size, stride)
    g_patches, _, _ = extract_patches_with_padding(gabor_img, patch_size, stride)
    h_pad, w_pad = padded_shape

    accum = np.zeros((h_pad, w_pad, NUM_CLASSES), dtype=np.float32)
    count_mask = np.zeros((h_pad, w_pad), dtype=np.float32)

    for (x, y), e, g in zip(positions, e_patches, g_patches):
        input_patch = np.stack([e / 255.0, g / 255.0], axis=-1)[np.newaxis, ...]
        pred = model.predict(input_patch, verbose=0)[0]  # shape: [256, 256, NUM_CLASSES]
        accum[y:y+patch_size[1], x:x+patch_size[0], :] += pred
        count_mask[y:y+patch_size[1], x:x+patch_size[0]] += 1

    count_mask = count_mask[..., np.newaxis]
    count_mask[count_mask == 0] = 1
    mean_probs = accum / count_mask
    final_mask = np.argmax(mean_probs, axis=-1).astype(np.uint8)
    return final_mask[:edge_img.shape[0], :edge_img.shape[1]]

def compute_metrics(y_true, y_pred, num_classes):
    mask = y_true != 255
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    cm = confusion_matrix(y_true.flatten(), y_pred.flatten(), labels=list(range(num_classes)))
    ious = []
    for i in range(num_classes):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        union = TP + FP + FN
        ious.append(TP / union if union > 0 else np.nan)
    acc = (y_true == y_pred).sum() / len(y_true)
    return acc, np.nanmean(ious), ious

def colorize_mask(mask):
    colormap = np.array([
        [0, 0, 0],         # 0: background or class 0
        [127, 127, 127],   # 1: class 1
        [255, 0, 0],       # 2: class 2
        [0, 255, 0],       # 3: class 3
    ], dtype=np.uint8)
    result = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for v in np.unique(mask):
        if v < len(colormap):
            result[mask == v] = colormap[v]
    return result

def save_comparison_image(gt, pred, out_path):
    gt_color = colorize_mask(gt)
    pred_color = colorize_mask(pred)
    concat = np.concatenate((gt_color, pred_color), axis=1)
    cv2.imwrite(out_path, concat)

# --- TRAINING ---
print("Loading training data...")
X_train, Y_train = load_patched_dataset(TRAIN_EDGE, TRAIN_GABOR, TRAIN_LABEL)
print(f"Loaded {len(X_train)} training patches.")

print("Building model...")
model = get_unet()
model.compile(optimizer='adam', loss=masked_categorical_crossentropy, metrics=['accuracy'])

print("Training model...")
model.fit(X_train, Y_train, validation_split=0.2, epochs=EPOCHS,
          batch_size=BATCH_SIZE, callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])

# --- TESTING ---
print("Evaluating on test images...")
metrics_list = []

for fname in sorted(os.listdir(TEST_EDGE)):
    try:
        edge = load_image(os.path.join(TEST_EDGE, fname))
        gabor = load_image(os.path.join(TEST_GABOR, fname))
        label = preprocess_label(load_image(os.path.join(TEST_LABEL, fname)))

        if edge.shape != gabor.shape or edge.shape != label.shape:
            print(f"Skipping {fname}: shape mismatch.")
            continue

        pred = predict_full_image(model, edge, gabor, PATCH_SIZE, STRIDE)
        acc, miou, ious = compute_metrics(label, pred, NUM_CLASSES)

        cv2.imwrite(os.path.join(RESULT_DIR, fname), pred)
        cv2.imwrite(os.path.join(COLOR_DIR, fname), colorize_mask(pred))
        save_comparison_image(label, pred, os.path.join(COMPARE_DIR, fname))

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
        print(f"Error processing {fname}: {e}")

# --- SAVE RESULTS ---
df = pd.DataFrame(metrics_list)
df.to_csv(os.path.join(BASE_PATH, "results/metrics.csv"), index=False)
print("Saved metrics to results/metrics.csv")
