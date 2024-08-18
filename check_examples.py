import pandas as pd

dataset = pd.read_csv("./InSDN/mergedCSV.csv")

count_labels = dataset["Label"].value_counts()
print(count_labels)
