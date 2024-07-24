import pandas as pd
import time
import json
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score, train_test_split

dataset = pd.read_csv("./InSDN/mergedCSV.csv")

X = dataset.drop("Label", axis=1)
Y = dataset["Label"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# normalise data

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
X_train_scaled = min_max_scaler.fit_transform(X_train)
X_test_scaled = min_max_scaler.transform(X_test)

# fit to classifier

classifier = RandomForestClassifier(n_estimators=200)
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

# checking for overfitting with train dataset

training_predictions = classifier.predict(X_train_scaled)
train_accuracy = metrics.accuracy_score(Y_train, training_predictions)

# cross validation

cross_val = cross_val_score(classifier, X_train_scaled, Y_train, cv=5)

# write metrics to obj

metrics_obj = {
    "Accuracy": accuracy,
    "Recall": recall,
    "Precision": precision,
    "F1 Score": f1_score,
    "Training Time": training_time
}

# write metrics to file

with open("./ML/RandomForest/optimizedMetrics.json", "w") as json_file:
    json.dump(metrics_obj, json_file, indent=4)

print("Metrics written to file in folder")
print("Training accuracy:", train_accuracy, "Testing accuracy:", accuracy)
print(f'Cross-validation scores: {cross_val}')
print(f'Mean CV Score: {cross_val.mean()}')