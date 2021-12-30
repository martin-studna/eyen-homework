#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.keras import layers
import argparse
import cv2 as cv
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--files', nargs='+', default=["test.jpeg", "test2.jpg"])

WIDTH, HEIGHT, CHANNELS = 1536, 2048, 3


IMG_SIZE = 224

inputs = layers.Input(shape=(WIDTH, HEIGHT, CHANNELS))
x = layers.Resizing(IMG_SIZE, IMG_SIZE)(inputs)


model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_tensor=x
)
model.trainable = False


x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
x = layers.BatchNormalization()(x)
top_dropout_rate = 0.2
x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
outputs = layers.Dense(1, name="pred")(x)


model = tf.keras.Model(inputs=[inputs], outputs=[outputs])


model.load_weights("./car_couplings.h5")

args = parser.parse_args()
for f in args.files:
    if f is not None:
        image = cv.imread(f)
        pred = model.predict(np.array([image]))[0,0]
        print(pred)
