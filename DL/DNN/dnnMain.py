import pandas as pd
import json
import time

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

from keras.callbacks import Callback, EarlyStopping
# from keras.utils import plot_model

dataset = pd.read_csv("./InSDN/mergedCSV.csv")

X = dataset.drop("Label", axis=1)
Y = dataset["Label"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# normalise data

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
X_train_scaled = min_max_scaler.fit_transform(X_train)
X_test_scaled = min_max_scaler.transform(X_test)

# create callback

class training_time_callback(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_start_time)

# add early stopping callback

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    verbose=1,
    restore_best_weights=True,
    start_from_epoch=3
)

# callback taken from callback file
cb = training_time_callback()

# build model using dense layers (4 layer)

def model_build(input_shape):
    model = Sequential()

    model.add(Dense(128, activation="relu", input_shape=input_shape))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.5))

# adding 3 more layers
    model.add(Dense(16, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation="relu"))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation="sigmoid"))

    return model

input_shape = (X_train_scaled.shape[1],)
dnn_model = model_build(input_shape)

# compile  model

dnn_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy","recall", "precision", "f1_score"])

# train model
dnn_model.fit(x=X_train_scaled, y=Y_train, epochs=30, validation_data=(X_test_scaled, Y_test), callbacks=[cb, early_stopping])

## evaluate model & return metrics

loss, accuracy, recall, precision, f1_score = dnn_model.evaluate(X_test_scaled, Y_test)
training_time = sum(cb.times)

metrics_obj = {
    "Accuracy": accuracy,
    "Recall": recall,
    "Precision": precision,
    "F1 Score": f1_score,
    "Training Time": training_time
}

# write metrics to file

with open("./DL/DNN/metricsWithDropoutSeveralLayers2.json", "w") as json_file:
    json.dump(metrics_obj, json_file, indent=4)

print("Metrics written to file in folder")

# write model to file

# dnn_model_png = 'DL/DNN/dnn_model_1.png'
# plot_model(model=dnn_model, to_file=dnn_model_png, show_shapes=True, show_layer_activations=True)