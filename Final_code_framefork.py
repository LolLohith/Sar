import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

IMG_SIZE = (256, 256)
NUM_CLASSES = 4
BATCH_SIZE = 8
EPOCHS = 20

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
        inputs, seg_labels, edge_labels, gabor_labels = [], [], [], []

        for i in idxs:
            sar = cv2.imread(self.x_paths[i], cv2.IMREAD_GRAYSCALE)
            sar = cv2.resize(sar, IMG_SIZE).astype(np.float32) / 255.0
            sar = np.expand_dims(sar, axis=-1)

            edge = cv2.imread(self.edge_paths[i], cv2.IMREAD_GRAYSCALE)
            edge = cv2.resize(edge, IMG_SIZE)
            edge = (edge > 127).astype(np.float32)
            edge = np.expand_dims(edge, axis=-1)

            gabor = cv2.imread(self.gabor_paths[i], cv2.IMREAD_GRAYSCALE)
            gabor = cv2.resize(gabor, IMG_SIZE).astype(np.float32) / 255.0
            gabor = np.expand_dims(gabor, axis=-1)

            combined_input = np.concatenate([edge, sar, gabor], axis=-1)

            gt = cv2.imread(self.gt_paths[i], cv2.IMREAD_GRAYSCALE)
            gt = cv2.resize(gt, IMG_SIZE)

            gabor_label = cv2.imread(self.gabor_paths[i], cv2.IMREAD_GRAYSCALE)
            gabor_label = cv2.resize(gabor_label, IMG_SIZE)
            gabor_label = (gabor_label * NUM_CLASSES) // 256

            inputs.append(combined_input)
            seg_labels.append(gt)
            edge_labels.append(edge.squeeze())
            gabor_labels.append(gabor_label)

        return np.array(inputs), {
            "seg_head": np.expand_dims(np.array(seg_labels), -1),
            "edge_head": np.expand_dims(np.array(edge_labels), -1),
            "gabor_head": np.expand_dims(np.array(gabor_labels), -1),
        }

def build_model(input_shape=(256, 256, 3), num_classes=4):
    inp = Input(shape=input_shape)
    x = tf.keras.applications.ResNet50(include_top=False, weights=None, input_tensor=inp).output
    x = GlobalAveragePooling2D()(x)
    shared = Dense(256, activation='relu')(x)

    seg = Dense(256, activation='relu')(shared)
    seg = Dense(256*256*num_classes, activation='softmax')(seg)
    seg = Reshape((256, 256, num_classes), name="seg_head")(seg)

    edge = Dense(256, activation='relu')(shared)
    edge = Dense(256*256, activation='sigmoid')(edge)
    edge = Reshape((256, 256, 1), name="edge_head")(edge)

    gabor = Dense(256, activation='relu')(shared)
    gabor = Dense(256*256*num_classes, activation='softmax')(gabor)
    gabor = Reshape((256, 256, num_classes), name="gabor_head")(gabor)

    return Model(inputs=inp, outputs=[seg, edge, gabor])

def load_paths(in_dir, gt_dir, edge_dir, gabor_dir):
    names = sorted(os.listdir(in_dir))
    x_paths = [os.path.join(in_dir, f) for f in names]
    gt_paths = [os.path.join(gt_dir, f.replace(".png", "_mask.png")) for f in names]
    edge_paths = [os.path.join(edge_dir, f.replace(".png", "_edge.png")) for f in names]
    gabor_paths = [os.path.join(gabor_dir, f.replace(".png", "_gabor_labelled.png")) for f in names]
    return x_paths, gt_paths, edge_paths, gabor_paths

def train():
    input_dir = "dataset/Denoised"
    gt_dir = "dataset/GroundTruth_Masks"
    edge_dir = "dataset/Edge_Labels"
    gabor_dir = "dataset/Gabor_Labelled"

    x_paths, gt_paths, edge_paths, gabor_paths = load_paths(input_dir, gt_dir, edge_dir, gabor_dir)
    train_gen = MultiTaskDataGen(x_paths, gt_paths, edge_paths, gabor_paths)

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

    model.fit(train_gen, epochs=EPOCHS, callbacks=[
        ModelCheckpoint("multitask_model.h5", save_best_only=True)
    ])

def test():
    model = tf.keras.models.load_model("multitask_model.h5", custom_objects={"iou_metric": iou_metric})

    test_dir = "dataset/test_images"
    gt_dir = "dataset/GroundTruth_Masks"
    edge_dir = "dataset/Edge_Labels"
    gabor_dir = "dataset/Gabor_Labelled"
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "Comparison_All"), exist_ok=True)

    label_names = ['background', 'barren', 'vegetation', 'urban']

    for fname in sorted(os.listdir(test_dir)):
        base = os.path.splitext(fname)[0]
        sar = cv2.imread(os.path.join(test_dir, fname), cv2.IMREAD_GRAYSCALE)
        sar = cv2.resize(sar, IMG_SIZE).astype(np.float32) / 255.0
        sar = np.expand_dims(sar, axis=-1)

        edge = cv2.imread(os.path.join(edge_dir, base + "_edge.png"), cv2.IMREAD_GRAYSCALE)
        edge = cv2.resize(edge, IMG_SIZE)
        edge = (edge > 127).astype(np.float32)
        edge = np.expand_dims(edge, axis=-1)

        gabor = cv2.imread(os.path.join(gabor_dir, base + "_gabor_labelled.png"), cv2.IMREAD_GRAYSCALE)
        gabor = cv2.resize(gabor, IMG_SIZE).astype(np.float32) / 255.0
        gabor = np.expand_dims(gabor, axis=-1)

        inp = np.concatenate([edge, sar, gabor], axis=-1)[np.newaxis, ...]

        pred_seg, pred_edge, pred_gabor = model.predict(inp)
        pred_seg = np.argmax(pred_seg[0], axis=-1)
        pred_edge = (pred_edge[0, ..., 0] > 0.5).astype(np.uint8)
        pred_gabor = np.argmax(pred_gabor[0], axis=-1)

        gt = cv2.resize(cv2.imread(os.path.join(gt_dir, base + "_mask.png"), cv2.IMREAD_GRAYSCALE), IMG_SIZE)
        edge_gt = (edge > 0.5).astype(np.uint8)
        gabor_gt = cv2.imread(os.path.join(gabor_dir, base + "_gabor_labelled.png"), cv2.IMREAD_GRAYSCALE)
        gabor_gt = cv2.resize(gabor_gt, IMG_SIZE)
        gabor_gt = (gabor_gt * NUM_CLASSES) // 256

        acc_seg = accuracy_score(gt.flatten(), pred_seg.flatten())
        acc_edge = accuracy_score(edge_gt.flatten(), pred_edge.flatten())
        acc_gabor = accuracy_score(gabor_gt.flatten(), pred_gabor.flatten())

        print(f"[{fname}] Seg: {acc_seg:.3f}, Edge: {acc_edge:.3f}, Gabor: {acc_gabor:.3f}")
        dominant_class = np.bincount(pred_seg.flatten()).argmax()
        print(f"Predicted Dominant Class: {label_names[dominant_class]}")

        fig, axs = plt.subplots(1, 5, figsize=(16, 4))
        axs[0].imshow(sar.squeeze(), cmap='gray')
        axs[0].set_title("Input SAR")
        axs[1].imshow(pred_seg, cmap='jet')
        axs[1].set_title("Predicted Seg")
        axs[2].imshow(gt, cmap='jet')
        axs[2].set_title("GT Seg")
        axs[3].imshow(pred_edge, cmap='gray')
        axs[3].set_title("Predicted Edge")
        axs[4].imshow(pred_gabor, cmap='jet')
        axs[4].set_title("Predicted Gabor")
        for ax in axs:
            ax.axis('off')
        plt.savefig(os.path.join(save_dir, "Comparison_All", base + "_compare_all.png"))
        plt.close()

if __name__ == "__main__":
    train()
    test()
