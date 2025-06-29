
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Reshape, Layer

# --- Constants ---
IMG_SIZE = (256, 256)
NUM_CLASSES = 4
BATCH_SIZE = 8
EPOCHS = 20

# --- Custom Layers and Metrics ---

class GrayToRGB(Layer):
    def call(self, inputs):
        return tf.image.grayscale_to_rgb(inputs)

def iou_metric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, tf.int32)
    ious = []
    for i in range(NUM_CLASSES):
        intersection = tf.reduce_sum(tf.cast((y_true == i) & (y_pred == i), tf.float32))
        union = tf.reduce_sum(tf.cast((y_true == i) | (y_pred == i), tf.float32)) + 1e-7
        ious.append(intersection / union)
    return tf.reduce_mean(ious)

# --- Data Generator ---

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
            gabor = (gabor * NUM_CLASSES) // 256

            x_batch.append(img)
            gt_batch.append(gt)
            edge_batch.append(edge)
            gabor_batch.append(gabor)

        return np.array(x_batch), {
            "seg_head": np.expand_dims(np.array(gt_batch), -1),
            "edge_head": np.expand_dims(np.array(edge_batch), -1),
            "gabor_head": np.expand_dims(np.array(gabor_batch), -1),
        }

# --- Model Builder ---

def build_model(input_shape=(256, 256, 1), num_classes=4):
    inp = Input(shape=input_shape)
    x = GrayToRGB()(inp)
    base = ResNet50(include_top=False, weights='imagenet', input_tensor=x)
    x = base.output
    x = GlobalAveragePooling2D()(x)
    shared = Dense(256, activation='relu')(x)

    x_seg = Dense(256, activation='relu')(shared)
    x_seg = Dense(input_shape[0] * input_shape[1] * num_classes, activation='softmax')(x_seg)
    x_seg = Reshape((*input_shape[:2], num_classes), name="seg_head")(x_seg)

    x_edge = Dense(256, activation='relu')(shared)
    x_edge = Dense(input_shape[0] * input_shape[1], activation='sigmoid')(x_edge)
    x_edge = Reshape((*input_shape[:2], 1), name="edge_head")(x_edge)

    x_gabor = Dense(256, activation='relu')(shared)
    x_gabor = Dense(input_shape[0] * input_shape[1] * num_classes, activation='softmax')(x_gabor)
    x_gabor = Reshape((*input_shape[:2], num_classes), name="gabor_head")(x_gabor)

    return Model(inputs=inp, outputs=[x_seg, x_edge, x_gabor])

# --- Data Loader ---

def load_paths(in_dir, gt_dir, e_dir, g_dir):
    names = sorted(os.listdir(in_dir))
    x_paths = [os.path.join(in_dir, f) for f in names]
    gt_paths = [os.path.join(gt_dir, f.replace(".png", "_mask.png")) for f in names]
    e_paths = [os.path.join(e_dir, f.replace(".png", "_edge.png")) for f in names]
    g_paths = [os.path.join(g_dir, f.replace(".png", "_gabor_labelled.png")) for f in names]
    return x_paths, gt_paths, e_paths, g_paths

# --- Train ---

def train():
    input_dir = "dataset/Denoised"
    gt_dir = "dataset/GroundTruth_Masks"
    edge_dir = "dataset/Edge_Labels"
    gabor_dir = "dataset/Gabor_Labelled"

    x_paths, gt_paths, e_paths, g_paths = load_paths(input_dir, gt_dir, edge_dir, gabor_dir)
    train_gen = MultiTaskDataGen(x_paths, gt_paths, e_paths, g_paths, batch_size=BATCH_SIZE)

    model = build_model()
    model.compile(
        optimizer='adam',
        loss={
            "seg_head": "sparse_categorical_crossentropy",
            "edge_head": "binary_crossentropy",
            "gabor_head": "sparse_categorical_crossentropy"
        },
        metrics={
            "seg_head": ["accuracy", iou_metric],
            "edge_head": "accuracy",
            "gabor_head": "accuracy"
        }
    )

    history = model.fit(train_gen, epochs=EPOCHS, callbacks=[
        ModelCheckpoint("multitask_model.h5", save_best_only=True)
    ])

# --- Test ---

def test():
    model = tf.keras.models.load_model("multitask_model.h5", custom_objects={"GrayToRGB": GrayToRGB, "iou_metric": iou_metric})

    test_dir = "dataset/test_images"
    gt_dir = "dataset/GroundTruth_Masks"
    edge_dir = "dataset/Edge_Labels"
    gabor_dir = "dataset/Gabor_Labelled"
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)

    for fname in sorted(os.listdir(test_dir)):
        base = os.path.splitext(fname)[0]
        img_path = os.path.join(test_dir, fname)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, IMG_SIZE)
        inp = img_resized[np.newaxis, ..., np.newaxis] / 255.0

        pred_seg, pred_edge, pred_gabor = model.predict(inp)
        pred_seg = np.argmax(pred_seg[0], axis=-1)
        pred_edge = (pred_edge[0, ..., 0] > 0.5).astype(np.uint8)
        pred_gabor = np.argmax(pred_gabor[0], axis=-1)

        gt = cv2.resize(cv2.imread(os.path.join(gt_dir, base + "_mask.png"), cv2.IMREAD_GRAYSCALE), IMG_SIZE)
        edge_gt = cv2.resize(cv2.imread(os.path.join(edge_dir, base + "_edge.png"), cv2.IMREAD_GRAYSCALE), IMG_SIZE)
        edge_gt = (edge_gt > 127).astype(np.uint8)
        gabor_gt = cv2.imread(os.path.join(gabor_dir, base + "_gabor_labelled.png"))
        gabor_gt = cv2.cvtColor(cv2.resize(gabor_gt, IMG_SIZE), cv2.COLOR_BGR2GRAY)
        gabor_gt = (gabor_gt * NUM_CLASSES) // 256

        acc_seg = accuracy_score(gt.flatten(), pred_seg.flatten())
        acc_edge = accuracy_score(edge_gt.flatten(), pred_edge.flatten())
        acc_gabor = accuracy_score(gabor_gt.flatten(), pred_gabor.flatten())
        print(f"[{fname}] Seg: {acc_seg:.3f}, Edge: {acc_edge:.3f}, Gabor: {acc_gabor:.3f}")

if __name__ == "__main__":
    train()
    test()
