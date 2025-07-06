
# ✅ Enhanced DeepLabV3+ Multitask Training Script

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Conv2D, UpSampling2D, BatchNormalization, Activation
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

IMG_SIZE = (256, 256)
NUM_CLASSES = 4  # 0=Background, 1=Barren, 2=Vegetation, 3=Urban
BATCH_SIZE = 8
EPOCHS = 20

class MultiTaskDataGen(Sequence):
    def __init__(self, sar_paths, gt_paths, edge_paths, gabor_paths, batch_size=BATCH_SIZE, shuffle=True):
        self.sar_paths = sar_paths
        self.gt_paths = gt_paths
        self.edge_paths = edge_paths
        self.gabor_paths = gabor_paths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.sar_paths))
        self.on_epoch_end()

    def __len__(self):
        return len(self.sar_paths) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, idx):
        idxs = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        x_batch, y_batch = [], []

        for i in idxs:
            sar = cv2.imread(self.sar_paths[i], cv2.IMREAD_GRAYSCALE)
            sar = cv2.resize(sar, IMG_SIZE).astype(np.float32) / 255.0
            sar = np.expand_dims(sar, axis=-1)

            edge = cv2.imread(self.edge_paths[i], cv2.IMREAD_GRAYSCALE)
            edge = cv2.resize(edge, IMG_SIZE)
            edge = (edge > 127).astype(np.float32)
            edge = np.expand_dims(edge, axis=-1)

            gabor = cv2.imread(self.gabor_paths[i], cv2.IMREAD_GRAYSCALE)
            gabor = cv2.resize(gabor, IMG_SIZE).astype(np.float32) / 255.0
            gabor = np.expand_dims(gabor, axis=-1)

            stacked_input = np.concatenate([edge, sar, gabor], axis=-1)

            gt = cv2.imread(self.gt_paths[i], cv2.IMREAD_GRAYSCALE)
            gt = cv2.resize(gt, IMG_SIZE).astype(np.int32)

            x_batch.append(stacked_input)
            y_batch.append(gt)

        x_batch = np.array(x_batch)
        y_batch = np.expand_dims(np.array(y_batch), axis=-1)

        return x_batch, y_batch

def simple_deeplabv3plus(input_shape=(256, 256, 3), num_classes=4):
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    x = base_model.output
    x = UpSampling2D(size=(4, 4), interpolation='bilinear')(x)
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(num_classes, (1, 1), activation='softmax')(x)
    x = UpSampling2D(size=(4, 4), interpolation='bilinear')(x)
    model = Model(inputs=base_model.input, outputs=x)
    return model

def main():
    # Mock example paths
    sar_paths = sorted([f"data/sar/{f}" for f in os.listdir("data/sar")])
    gt_paths = sorted([f"data/labels/{f}" for f in os.listdir("data/labels")])
    edge_paths = sorted([f"data/edge/{f}" for f in os.listdir("data/edge")])
    gabor_paths = sorted([f"data/gabor/{f}" for f in os.listdir("data/gabor")])

    x_train, x_val, y_train, y_val, e_train, e_val, g_train, g_val = train_test_split(
        sar_paths, gt_paths, edge_paths, gabor_paths, test_size=0.2, random_state=42)

    train_gen = MultiTaskDataGen(x_train, y_train, e_train, g_train)
    val_gen = MultiTaskDataGen(x_val, y_val, e_val, g_val)

    model = simple_deeplabv3plus(input_shape=(256, 256, 3), num_classes=NUM_CLASSES)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

    model.save("urban_segmentation_model.h5")

if __name__ == "__main__":
    main()
