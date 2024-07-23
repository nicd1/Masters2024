import pandas as pd
import numpy as np
import json

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import time
from keras.callbacks import Callback

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# from callbacks import training_time_callback

dataset = pd.read_csv("./InSDN/mergedCSV.csv")

X = dataset.drop("Label", axis=1)
Y = dataset["Label"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# normalise data

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
X_train_scaled = min_max_scaler.fit_transform(X_train)
X_test_scaled = min_max_scaler.transform(X_test)

X_train_shaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_shaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# build model

def model_build(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation="relu", padding="same", input_shape=(input_shape)))
    model.add(MaxPooling1D(pool_size=1))

    model.add(Conv1D(filters=128, kernel_size=3, activation="relu", padding="same"))
    model.add(MaxPooling1D(pool_size=1))

    model.add(Conv1D(filters=256, kernel_size=3, activation="relu", padding="same"))
    model.add(MaxPooling1D(pool_size=1))

    model.add(Flatten())

    model.add(Dense(128, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    return model

# re-shape data for cnn
input_shape = (1, X_train_scaled.shape[1],)

# compile model

cnn_model = model_build(input_shape)

cnn_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy","recall", "precision", "f1_score"])

# create callback

class training_time_callback(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_start_time)

# train model

cb = training_time_callback()
cnn_model.fit(x=X_train_shaped, y=Y_train, epochs=10, validation_data=(X_test_shaped, Y_test), callbacks=[cb])


loss, accuracy, recall, precision, f1_score = cnn_model.evaluate(X_test_shaped, Y_test)
training_time = sum(cb.times)

metrics_obj = {
    "Accuracy": accuracy,
    "Recall": recall,
    "Precision": precision,
    "F1 Score": f1_score,
    "Training Time": training_time
}

# write metrics to file

with open("./DL/CNN/metrics.json", "w") as json_file:
    json.dump(metrics_obj, json_file, indent=4)

print("Metrics written to file in folder")
