import pandas as pd
import time
import json

from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.datasets import fetch_kddcup99
from sklearn.model_selection import train_test_split

kdd99cup = fetch_kddcup99(subset=None, download_if_missing=True, as_frame=True)

dataset = kdd99cup.frame

# malicious_labels = dataset[dataset['labels'] != b'normal.']

socket_information = ["protocol_type", "service", "flag"]

# remove examples with missing values

clean_dataset = dataset.dropna()

# remove duplicate examples

clean_dataset = clean_dataset.drop_duplicates()

# remove PII and socket information

clean_dataset = clean_dataset.drop(columns=socket_information)

# # one-hot encoding

clean_dataset['labels'] = clean_dataset['labels'].apply(lambda x: 0 if x == b'normal.' else 1)

print(f"Shape before cleaning: {dataset.shape}") # 494021, 42
print(f'Shape after cleaning: {clean_dataset.shape}') # 145586, 39

# split into training and testing

X = clean_dataset.drop("labels", axis=1)
Y = clean_dataset["labels"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# normalise data

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
X_train_scaled = min_max_scaler.fit_transform(X_train)
X_test_scaled = min_max_scaler.transform(X_test)

# fit to classifier

classifier = RandomForestClassifier()
start = time.time()
classifier.fit(X_train_scaled, Y_train)
stop = time.time()

#predictions using metrics

predictions = classifier.predict(X_test_scaled)

accuracy = metrics.accuracy_score(Y_test, predictions)
recall = metrics.recall_score(Y_test, predictions)
precision = metrics.precision_score(Y_test, predictions)
f1_score = metrics.f1_score(Y_test, predictions)
training_time = stop - start

# write metrics to obj

metrics_obj = {
    "Accuracy": accuracy,
    "Recall": recall,
    "Precision": precision,
    "F1 Score": f1_score,
    "Training Time": training_time
}

# write metrics to file

with open("./KDD99Cup/kddRfmetrics.json", "w") as json_file:
    json.dump(metrics_obj, json_file, indent=4)

print("Metrics written to file in folder")

