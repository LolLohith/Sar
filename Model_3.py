import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Reshape, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import Sequence
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# --- Constants ---
IMG_SIZE = (256, 256)
NUM_CLASSES = 4            # Classes: background, barren, vegetation, urban
NUM_GABOR_CLASSES = 4
BATCH_SIZE = 8
EPOCHS = 20

# --- Utility Classes and Functions ---

class GrayToRGB(Layer):
    def call(self, inputs):
        # Converts a grayscale image to an RGB image by replication
        return tf.image.grayscale_to_rgb(inputs)

class LRTensorBoard(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        if hasattr(lr, 'numpy'):
            lr = lr.numpy()
        print(f"Epoch {epoch+1}: Learning Rate = {lr}")

def iou_metric(y_true, y_pred):
    """
    Computes the mean Intersection Over Union (IoU) across all classes.
    y_true: ground truth mask [batch, H, W, 1]
    y_pred: predicted probabilities [batch, H, W, NUM_CLASSES]
    """
    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, tf.int32)
    ious = []
    for i in range(NUM_CLASSES):
        # Compute intersection and union for class i
        intersection = tf.reduce_sum(tf.cast((y_true == i) & (y_pred == i), tf.float32))
        union = tf.reduce_sum(tf.cast((y_true == i) | (y_pred == i), tf.float32)) + 1e-7
        ious.append(intersection / union)
    return tf.reduce_mean(ious)

class MultiTaskDataGen(Sequence):
    """
    Data generator that yields a batch of images along with three types of labels:
      - 'seg_head': Ground truth segmentation mask (4 classes)
      - 'edge_head': Binary edge mask
      - 'gabor_head': Gabor-labelled image (quantized to 4 classes)
    """
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
            # Load input image (assumed to be grayscale)
            img = cv2.imread(self.x_paths[i], cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, IMG_SIZE)
            img = np.expand_dims(img, axis=-1) / 255.0

            # Ground truth segmentation mask (should be 4-class labels)
            gt = cv2.imread(self.gt_paths[i], cv2.IMREAD_GRAYSCALE)
            gt = cv2.resize(gt, IMG_SIZE)

            # Edge mask (binary)
            edge = cv2.imread(self.edge_paths[i], cv2.IMREAD_GRAYSCALE)
            edge = cv2.resize(edge, IMG_SIZE)
            edge = (edge > 127).astype(np.uint8)

            # Gabor-labelled image (convert to single channel and quantize to range 0 to NUM_GABOR_CLASSES-1)
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

def build_model(input_shape=(256, 256, 1), num_classes=4, gabor_classes=4):
    """
    Builds a multitask segmentation model with three outputs:
      - seg_head: Predicts segmentation into num_classes
      - edge_head: Predicts binary edge mask
      - gabor_head: Predicts gabor-like segmentation with gabor_classes classes
    """
    inp = Input(shape=input_shape)
    x = GrayToRGB()(inp)
    # Use ResNet50 as backbone; note that we use weights from ImageNet.
    base = ResNet50(include_top=False, weights='imagenet', input_tensor=x)
    x = base.output
    x = GlobalAveragePooling2D()(x)
    shared = Dense(256, activation='relu')(x)

    # Segmentation head (4 classes with softmax)
    x_seg = Dense(256, activation='relu')(shared)
    x_seg = Dense(input_shape[0] * input_shape[1] * num_classes, activation='softmax')(x_seg)
    x_seg = Reshape((*input_shape[:2], num_classes), name="seg_head")(x_seg)

    # Edge detection head (binary mask with sigmoid)
    x_edge = Dense(256, activation='relu')(shared)
    x_edge = Dense(input_shape[0] * input_shape[1], activation='sigmoid')(x_edge)
    x_edge = Reshape((*input_shape[:2], 1), name="edge_head")(x_edge)

    # Gabor head (4 classes with softmax)
    x_gabor = Dense(256, activation='relu')(shared)
    x_gabor = Dense(input_shape[0] * input_shape[1] * gabor_classes, activation='softmax')(x_gabor)
    x_gabor = Reshape((*input_shape[:2], gabor_classes), name="gabor_head")(x_gabor)

    return Model(inputs=inp, outputs=[x_seg, x_edge, x_gabor])

def load_paths(in_dir, gt_dir, e_dir, g_dir):
    """
    Constructs lists of file paths for input images and labels.
    Assumes the same file names in the input directory for all associated files,
    and replaces the extension to get label file names:
       - Ground truth: replace '.png' with '_mask.png'
       - Edge: replace '.png' with '_edge.png'
       - Gabor: replace '.png' with '_gabor_labelled.png'
    """
    names = sorted(os.listdir(in_dir))
    x_paths = [os.path.join(in_dir, f) for f in names]
    gt_paths = [os.path.join(gt_dir, f.replace(".png", "_mask.png")) for f in names]
    e_paths = [os.path.join(e_dir, f.replace(".png", "_edge.png")) for f in names]
    g_paths = [os.path.join(g_dir, f.replace(".png", "_gabor_labelled.png")) for f in names]
    return x_paths, gt_paths, e_paths, g_paths

# --- Training Function ---
def train():
    # Define the folder paths for training data.
    input_dir = "dataset/Denoised"
    gt_dir = "dataset/GroundTruth_Masks"
    edge_dir = "dataset/Edge_Labels"
    gabor_dir = "dataset/Gabor_Labelled"

    # Load the file paths
    x_paths, gt_paths, e_paths, g_paths = load_paths(input_dir, gt_dir, edge_dir, gabor_dir)
    train_gen = MultiTaskDataGen(x_paths, gt_paths, e_paths, g_paths, batch_size=BATCH_SIZE)

    # Build the model
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

    # Train the model and save the best model.
    history = model.fit(train_gen, epochs=EPOCHS, callbacks=[
        ModelCheckpoint("multitask_segmentation.h5", save_best_only=True),
        LRTensorBoard()
    ])

    # Plot and save the loss curves.
    plt.figure()
    plt.plot(history.history['loss'], label='Global Loss')
    plt.plot(history.history['seg_head_loss'], label='Segmentation Loss')
    plt.plot(history.history['edge_head_loss'], label='Edge Loss')
    plt.plot(history.history['gabor_head_loss'], label='Gabor Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title("Loss Over Time")
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/Loss_Plot.png")
    plt.close()

# --- Testing Function ---
def test():
    # Load the trained model.
    custom_objs = {"GrayToRGB": GrayToRGB, "iou_metric": iou_metric}
    model = tf.keras.models.load_model("multitask_segmentation.h5", custom_objects=custom_objs)

    # Define folder paths for testing.
    test_dir = "dataset/test_images"
    gt_dir = "dataset/GroundTruth_Masks"
    edge_dir = "dataset/Edge_Labels"
    gabor_dir = "dataset/Gabor_Labelled"
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "Predicted_Segmentation"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "Predicted_Edge"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "Predicted_Gabor"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "Comparison_Segmentation"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "Comparison_All"), exist_ok=True)

    label_names = ['background', 'barren', 'vegetation', 'urban']

    # Process each test image.
    for fname in sorted(os.listdir(test_dir)):
        base = os.path.splitext(fname)[0]
        img_path = os.path.join(test_dir, fname)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, IMG_SIZE)
        inp = img_resized[np.newaxis, ..., np.newaxis] / 255.0

        # Get model predictions.
        pred_seg, pred_edge, pred_gabor = model.predict(inp)
        pred_seg = np.argmax(pred_seg[0], axis=-1)
        pred_edge = (pred_edge[0, ..., 0] > 0.5).astype(np.uint8)
        pred_gabor = np.argmax(pred_gabor[0], axis=-1)

        # Load ground truth images.
        gt = cv2.imread(os.path.join(gt_dir, base + "_mask.png"), cv2.IMREAD_GRAYSCALE)
        gt = cv2.resize(gt, IMG_SIZE)
        edge_gt = cv2.imread(os.path.join(edge_dir, base + "_edge.png"), cv2.IMREAD_GRAYSCALE)
        edge_gt = cv2.resize(edge_gt, IMG_SIZE)
        edge_gt = (edge_gt > 127).astype(np.uint8)
        gabor_gt = cv2.imread(os.path.join(gabor_dir, base + "_gabor_labelled.png"))
        gabor_gt = cv2.resize(cv2.cvtColor(gabor_gt, cv2.COLOR_BGR2GRAY), IMG_SIZE)
        gabor_gt = (gabor_gt * NUM_GABOR_CLASSES) // 256

        # Calculate accuracy scores.
        acc_seg = accuracy_score(gt.flatten(), pred_seg.flatten())
        acc_edge = accuracy_score(edge_gt.flatten(), pred_edge.flatten())
        acc_gabor = accuracy_score(gabor_gt.flatten(), pred_gabor.flatten())
        print(f"[{fname}] Seg: {acc_seg:.3f}, Edge: {acc_edge:.3f}, Gabor: {acc_gabor:.3f}")

        # Determine the dominant class in the predicted segmentation.
        pred_label = np.bincount(pred_seg.flatten()).argmax()
        print(f"Predicted Dominant Class: {label_names[pred_label]}")

        # Colorize segmentation maps using a colormap.
        seg_color = cv2.applyColorMap((pred_seg * 60).astype(np.uint8), cv2.COLORMAP_JET)
        gabor_color = cv2.applyColorMap((pred_gabor * 85).astype(np.uint8), cv2.COLORMAP_JET)
        gt_color = cv2.applyColorMap((gt * 60).astype(np.uint8), cv2.COLORMAP_JET)

        # Save predicted segmentation, edge, and gabor maps.
        cv2.imwrite(os.path.join(save_dir, "Predicted_Segmentation", base + "_pred_seg.png"), seg_color)
        cv2.imwrite(os.path.join(save_dir, "Predicted_Edge", base + "_pred_edge.png"), (pred_edge * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(save_dir, "Predicted_Gabor", base + "_pred_gabor.png"), gabor_color)

        # Save a side-by-side comparison of predicted and ground truth segmentation.
        seg_cmp = np.hstack((seg_color, gt_color))
        cv2.imwrite(os.path.join(save_dir, "Comparison_Segmentation", base + "_seg_cmp.png"), seg_cmp)

        # Save a full 5-panel comparison: input, predicted seg, ground truth seg, predicted edge, and predicted gabor.
        fig, axs = plt.subplots(1, 5, figsize=(16, 4))
        axs[0].imshow(img_resized, cmap='gray')
        axs[0].set_title("Input")
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
    # First, train the model.
    train()

    # Then, test the model on test images.
    test()
