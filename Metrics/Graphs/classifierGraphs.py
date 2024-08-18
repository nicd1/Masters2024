import json
from graphFunctions import create_simple_bar_graph

training_time = "Training Time"
# SVM

## Linear SVC

with open("ML/SVM/linearSVCMetrics.json", 'r') as file:
    linear_svc_data = json.load(file)

# removing training time from json

if training_time in linear_svc_data:
    linear_svc_training_time_data = linear_svc_data.pop(training_time)

values_linear_svc = list(linear_svc_data.values())

x_linear_svc = list(linear_svc_data.keys())
y_linear_svc = [round (value * 100, 4) for value in values_linear_svc]

create_simple_bar_graph(x=x_linear_svc, y=y_linear_svc, value_labels=True, colour="Red", title="SVM (Linear SVC) Metrics", save_path="Metrics/Graphs/pngs/linearSvcMetrics.png", dpi=400)

## Linear SVM Kernel

with open("ML/SVM/metrics.json", 'r') as file:
    linear_kernel_data = json.load(file)

# removing training time from json

if training_time in linear_kernel_data:
    linear_kernel_training_time_data = linear_kernel_data.pop(training_time)

values_lk = list(linear_kernel_data.values())

x_lk = list(linear_kernel_data.keys())
y_lk = [round (value * 100, 4) for value in values_lk]

create_simple_bar_graph(x=x_lk, y=y_lk, value_labels=True, colour="Red", title="Pre-Optimisation (Linear Kernel) SVM Metrics", save_path="Metrics/Graphs/pngs/preOptimisedSvmKernelMetrics.png", dpi=400)

# Random Forest

with open("ML/RandomForest/optimizedMetrics.json", 'r') as file:
    rf_data = json.load(file)

# removing training time from json

if training_time in rf_data :
    rf_training_time_data = rf_data.pop(training_time)

values_rf = list(rf_data.values())

x_rf = list(rf_data.keys())
y_rf = [round (value * 100, 4) for value in values_rf]

create_simple_bar_graph(x=x_rf, y=y_rf, value_labels=True, colour="Red", title="Random Forest Metrics", save_path="Metrics/Graphs/pngs/rfMetrics.png", dpi=400)

# Random Forest Pre-Optimised

with open("ML/RandomForest/metrics.json", 'r') as file:
    pre_optimised_rf_data = json.load(file)

# removing training time from json

if training_time in pre_optimised_rf_data :
    pre_optimised_rf_training_time_data = pre_optimised_rf_data.pop(training_time)

pre_optimised_values_rf = list(pre_optimised_rf_data.values())

x_preop_rf = list(pre_optimised_rf_data.keys())
y_preop_rf = [round (value * 100, 4) for value in pre_optimised_values_rf]

create_simple_bar_graph(x=x_preop_rf, y=y_preop_rf, value_labels=True, colour="Red", title="Pre-Optimised Random Forest Metrics", save_path="Metrics/Graphs/pngs/preOptimisedRfMetrics.png", dpi=400)

# DNN

with open("DL/DNN/metricsWithDropout.json", 'r') as file:
    dnn_data = json.load(file)

# removing training time from json

if training_time in dnn_data :
    dnn_training_time_data = dnn_data.pop(training_time)

values_dnn = list(dnn_data.values())

x_dnn = list(dnn_data.keys())
y_dnn = [round (value * 100, 4) for value in values_dnn]

create_simple_bar_graph(x=x_dnn, y=y_dnn, value_labels=True, colour="Red", title="DNN Metrics", save_path="Metrics/Graphs/pngs/dnnMetrics.png", dpi=400)

# pre-optimised DNN

with open("DL/DNN/metrics.json", 'r') as file:
    pre_optimised_dnn_data = json.load(file)

# removing training time from json

if training_time in pre_optimised_dnn_data :
    pre_optimised_dnn_training_time_data = pre_optimised_dnn_data.pop(training_time)

pre_optimised_values_dnn = list(pre_optimised_dnn_data.values())

x_preop_dnn = list(pre_optimised_dnn_data.keys())
y_preop_dnn = [round (value * 100, 4) for value in pre_optimised_values_dnn]

create_simple_bar_graph(x=x_preop_dnn, y=y_preop_dnn, value_labels=True, colour="Red", title="Pre-Optimised DNN Metrics", save_path="Metrics/Graphs/pngs/preOptimisedDnnMetrics.png", dpi=400)

# CNN

with open("DL/CNN/optimizedMetricsWithDropout.json", 'r') as file:
    cnn_data = json.load(file)

# removing training time from json

if training_time in cnn_data :
    cnn_training_time_data = cnn_data.pop(training_time)

values_cnn = list(cnn_data.values())

x_cnn = list(cnn_data.keys())
y_cnn = [round (value * 100, 4) for value in values_cnn]

create_simple_bar_graph(x=x_cnn, y=y_cnn, value_labels=True, colour="Red", title="CNN Metrics", save_path="Metrics/Graphs/pngs/cnnMetrics.png", dpi=400)

# pre optimised CNN

with open("DL/CNN/metricsWithEarlyStopping.json", 'r') as file:
    pre_optimised_cnn_data = json.load(file)

# removing training time from json

if training_time in pre_optimised_cnn_data :
    pre_optimised_cnn_training_time_data = pre_optimised_cnn_data.pop(training_time)

pre_optimised_values_cnn = list(pre_optimised_cnn_data.values())

x_preop_cnn = list(pre_optimised_cnn_data.keys())
y_preop_cnn = [round (value * 100, 4) for value in pre_optimised_values_cnn]

create_simple_bar_graph(x=x_preop_cnn, y=y_preop_cnn, value_labels=True, colour="Red", title="Pre-Optimised CNN Metrics", save_path="Metrics/Graphs/pngs/preOptimisedCnnMetrics.png", dpi=400)

# Training Times graph

training_times = {
    "SVM" : linear_svc_training_time_data,
    "Random Forest" : rf_training_time_data,
    "DNN" : dnn_training_time_data,
    "CNN" : cnn_training_time_data
}

values_times = list(training_times.values())

x_times = list(training_times.keys())
y_times = [round(value, 2) for value in values_times]

create_simple_bar_graph(x=x_times, y=y_times, value_labels=True, colour="Blue", title="Training Times (seconds)", save_path="Metrics/Graphs/pngs/trainingTimesMetrics.png", dpi=400)

# Pre Optimised Training Times graph

pre_op_training_times = {
    "SVM " : linear_kernel_training_time_data,
    "Random Forest" : pre_optimised_rf_training_time_data,
    "DNN" : pre_optimised_dnn_training_time_data,
    "CNN" : pre_optimised_cnn_training_time_data
}

pre_optimised_values_times = list(pre_op_training_times.values())

x_preop_times = list(pre_op_training_times.keys())
y_preop_times = [round(value, 2) for value in pre_optimised_values_times]

create_simple_bar_graph(x=x_preop_times, y=y_preop_times, value_labels=True, colour="Blue", title="Pre-Optimised Training Times (seconds)", save_path="Metrics/Graphs/pngs/preOptimisedTrainingTimesMetrics.png", dpi=400)
