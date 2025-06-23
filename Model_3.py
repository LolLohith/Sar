import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Reshape, Layer
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# --- Constants ---
IMG_SIZE = (256, 256)
NUM_CLASSES = 4   # background, barren, vegetation, urban
NUM_GABOR_CLASSES = 4
BATCH_SIZE = 8
EPOCHS = 20

# --- Dataset Paths ---
base_dir = "dataset"
input_dir = os.path.join(base_dir, "Denoised")
gt_mask_dir = os.path.join(base_dir, "GroundTruth_Masks")
edge_label_dir = os.path.join(base_dir, "Edge_Labels")
gabor_label_dir = os.path.join(base_dir, "Gabor_Labelled")
test_dir = os.path.join(base_dir, "test_images")


# --- Convert grayscale image to RGB for ResNet ---
class GrayToRGB(Layer):
    def call(self, inputs):
        return tf.image.grayscale_to_rgb(inputs)


# --- Data Generator for Multi-task ---
class MultiTaskDataGen(Sequence):
    def __init__(self, x_paths, gt_paths, edge_paths, gabor_paths, batch_size=BATCH_SIZE, shuffle=True):
        self.x_paths = x_paths
        self.gt_paths = gt_paths
        self.edge_paths = edge_paths
        self.gabor_paths = gabor_paths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.x_paths))
        self.on_epoch_end()

    def __len__(self):
        return len(self.x_paths) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, idx):
        idxs = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        x_batch, gt_batch, edge_batch, gabor_batch = [], [], [], []

        for i in idxs:
            img = cv2.imread(self.x_paths[i], cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, IMG_SIZE)
            img = np.expand_dims(img, axis=-1) / 255.0

            gt = cv2.imread(self.gt_paths[i], cv2.IMREAD_GRAYSCALE)
            gt = cv2.resize(gt, IMG_SIZE)

            edge = cv2.imread(self.edge_paths[i], cv2.IMREAD_GRAYSCALE)
            edge = cv2.resize(edge, IMG_SIZE)
            edge = (edge > 127).astype(np.uint8)

            gabor = cv2.imread(self.gabor_paths[i])
            gabor = cv2.cvtColor(cv2.resize(gabor, IMG_SIZE), cv2.COLOR_BGR2GRAY)
            gabor = (gabor * NUM_GABOR_CLASSES) // 256

            x_batch.append(img)
            gt_batch.append(gt)
            edge_batch.append(edge)
            gabor_batch.append(gabor)

        return np.array(x_batch), {
            "seg_head": np.expand_dims(np.array(gt_batch), -1),
            "edge_head": np.expand_dims(np.array(edge_batch), -1),
            "gabor_head": np.expand_dims(np.array(gabor_batch), -1),
        }


# --- Load image paths ---
def load_paths(input_dir, gt_dir, edge_dir, gabor_dir):
    names = sorted(os.listdir(input_dir))
    x_paths = [os.path.join(input_dir, f) for f in names]
    gt_paths = [os.path.join(gt_dir, f.replace(".png", "_mask.png")) for f in names]
    e_paths = [os.path.join(edge_dir, f.replace(".png", "_edge.png")) for f in names]
    g_paths = [os.path.join(gabor_dir, f.replace(".png", "_gabor_labelled.png")) for f in names]
    return x_paths, gt_paths, e_paths, g_paths

train_x, train_gt, train_edge, train_gabor = load_paths(input_dir, gt_mask_dir, edge_label_dir, gabor_label_dir)


# --- Build Multi-output Model ---
def build_model(input_shape=(256, 256, 1), num_classes=4, gabor_classes=4):
    inp = Input(shape=input_shape)
    x = GrayToRGB()(inp)
    base = ResNet50(include_top=False, weights='imagenet', input_tensor=x)
    x = base.output
    x = GlobalAveragePooling2D()(x)
    shared = Dense(256, activation='relu')(x)

    # Ground Truth segmentation head
    x_seg = Dense(256, activation='relu')(shared)
    x_seg = Dense(input_shape[0] * input_shape[1] * num_classes, activation='softmax')(x_seg)
    x_seg = Reshape((*input_shape[:2], num_classes), name="seg_head")(x_seg)

    # Edge head
    x_edge = Dense(256, activation='relu')(shared)
    x_edge = Dense(input_shape[0] * input_shape[1], activation='sigmoid')(x_edge)
    x_edge = Reshape((*input_shape[:2], 1), name="edge_head")(x_edge)

    # Gabor head
    x_gabor = Dense(256, activation='relu')(shared)
    x_gabor = Dense(input_shape[0] * input_shape[1] * gabor_classes, activation='softmax')(x_gabor)
    x_gabor = Reshape((*input_shape[:2], gabor_classes), name="gabor_head")(x_gabor)

    return Model(inputs=inp, outputs=[x_seg, x_edge, x_gabor])


# --- Compile and Train ---
model = build_model()
model.compile(
    optimizer='adam',
    loss={
        "seg_head": "sparse_categorical_crossentropy",
        "edge_head": "binary_crossentropy",
        "gabor_head": "sparse_categorical_crossentropy"
    },
    metrics={
        "seg_head": "accuracy",
        "edge_head": "accuracy",
        "gabor_head": "accuracy"
    }
)

train_gen = MultiTaskDataGen(train_x, train_gt, train_edge, train_gabor)
model.fit(train_gen, epochs=EPOCHS, callbacks=[ModelCheckpoint("multitask_segmentation.h5", save_best_only=True)])


# --- Test & Visualize ---
def test_model(model, test_folder, gt_dir, edge_dir, gabor_dir):
    for fname in sorted(os.listdir(test_folder)):
        img = cv2.imread(os.path.join(test_folder, fname), cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, IMG_SIZE)
        input_tensor = img_resized[np.newaxis, ..., np.newaxis] / 255.0

        pred_seg, pred_edge, pred_gabor = model.predict(input_tensor)
        pred_seg = np.argmax(pred_seg[0], axis=-1)
        pred_edge = (pred_edge[0, ..., 0] > 0.5).astype(np.uint8)
        pred_gabor = np.argmax(pred_gabor[0], axis=-1)

        base = os.path.splitext(fname)[0]
        gt = cv2.imread(os.path.join(gt_dir, base + "_mask.png"), cv2.IMREAD_GRAYSCALE)
        gt = cv2.resize(gt, IMG_SIZE)

        edge_gt = cv2.imread(os.path.join(edge_dir, base + "_edge.png"), cv2.IMREAD_GRAYSCALE)
        edge_gt = cv2.resize(edge_gt, IMG_SIZE)
        edge_gt = (edge_gt > 127).astype(np.uint8)

        gabor_gt = cv2.imread(os.path.join(gabor_dir, base + "_gabor_labelled.png"))
        gabor_gt = cv2.resize(cv2.cvtColor(gabor_gt, cv2.COLOR_BGR2GRAY), IMG_SIZE)
        gabor_gt = (gabor_gt * NUM_GABOR_CLASSES) // 256

        # Accuracy
        acc_seg = accuracy_score(gt.flatten(), pred_seg.flatten())
        acc_edge = accuracy_score(edge_gt.flatten(), pred_edge.flatten())
        acc_gabor = accuracy_score(gabor_gt.flatten(), pred_gabor.flatten())
        print(f"[{fname}] Class Acc: {acc_seg:.3f}, Edge Acc: {acc_edge:.3f}, Gabor Acc: {acc_gabor:.3f}")

        # Dominant region classification
        pred_label = np.bincount(pred_seg.flatten()).argmax()
        label_names = ['background', 'barren', 'vegetation', 'urban']
        print(f"Predicted Region Class: {label_names[pred_label]}")

        # Visualization
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 5, 1); plt.imshow(img_resized, cmap='gray'); plt.title("Input")
        plt.subplot(1, 5, 2); plt.imshow(pred_seg, cmap='jet'); plt.title("Predicted Seg")
        plt.subplot(1, 5, 3); plt.imshow(gt, cmap='jet'); plt.title("GT Seg")
        plt.subplot(1, 5, 4); plt.imshow(pred_edge, cmap='gray'); plt.title("Edge")
        plt.subplot(1, 5, 5); plt.imshow(pred_gabor, cmap='jet'); plt.title("Gabor")
        plt.show()


# Run on test set
test_model(model, test_dir, gt_mask_dir, edge_label_dir, gabor_label_dir)
