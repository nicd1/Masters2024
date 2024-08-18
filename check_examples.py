import pandas as pd

dataset = pd.read_csv("./InSDN/mergedCSV.csv")

normal_count = dataset["Label"].value_counts()
print(normal_count)
