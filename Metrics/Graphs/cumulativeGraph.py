import json
from graphFunctions import create_grouped_bar_graph

with open("Metrics/concat_metrics.json", 'r') as file:
    concat_data = json.load(file)

classifiers = list(concat_data.keys())
metrics = list(concat_data.values())
accuracy  = [concat_data["Accuracy"] * 100 for concat_data in concat_data.values()]
recall = [concat_data["Recall"] * 100 for concat_data in concat_data.values()]
precision = [concat_data["Precision"] * 100 for concat_data in concat_data.values()]
f1_score = [concat_data["F1 Score"] * 100 for concat_data in concat_data.values()]

create_grouped_bar_graph(
    data=[accuracy, recall, precision, f1_score], 
    bar_labels=classifiers, 
    colours=['purple', 'orange', 'green', 'cyan'], 
    title="Cumulative Metrics for Models", 
    legend_labels=metrics[0], 
    x_label="Classifiers", 
    y_label="Metrics", 
    save_path="Metrics/Graphs/pngs/cumulatedMetrics.png", 
    dpi=800
    )