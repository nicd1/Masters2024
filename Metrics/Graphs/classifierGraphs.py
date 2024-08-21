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

create_simple_bar_graph(x=x_linear_svc, y=y_linear_svc, value_labels=True, colour="Red", title="Linear SVC Metrics (Model B)", save_path="Metrics/Graphs/pngs/linearSvcMetrics.png", dpi=400)

## Linear SVM Kernel

with open("ML/SVM/metrics.json", 'r') as file:
    linear_kernel_data = json.load(file)

# removing training time from json

if training_time in linear_kernel_data:
    linear_kernel_training_time_data = linear_kernel_data.pop(training_time)

values_lk = list(linear_kernel_data.values())

x_lk = list(linear_kernel_data.keys())
y_lk = [round (value * 100, 4) for value in values_lk]

create_simple_bar_graph(x=x_lk, y=y_lk, value_labels=True, colour="Red", title="Linear Kernel SVM Metrics", save_path="Metrics/Graphs/pngs/linearKernelSVMMetrics.png", dpi=400)

# Poly Kernel SVM

with open("ML/SVM/SVCPolyKernelMetrics.json", 'r') as file:
    poly_kernel_data = json.load(file)

# removing training time from json

if training_time in poly_kernel_data:
    poly_kernel_training_time_data = poly_kernel_data.pop(training_time)

values_pk = list(poly_kernel_data.values())

x_pk = list(poly_kernel_data.keys())
y_pk = [round (value * 100, 4) for value in values_pk]

create_simple_bar_graph(x=x_lk, y=y_lk, value_labels=True, colour="Red", title="Polynomial SVM Kernel (Model C)", save_path="Metrics/Graphs/pngs/SVCPolyKernelMetrics.png", dpi=400)

# RBF  Kernel SVM

with open("ML/SVM/SVCRbfKernelMetrics.json", 'r') as file:
    rbf_kernel_data = json.load(file)

# removing training time from json

if training_time in rbf_kernel_data:
    rbf_kernel_training_time_data = rbf_kernel_data.pop(training_time)

values_rbfk = list(rbf_kernel_data.values())

x_rbfk = list(rbf_kernel_data.keys())
y_rbfk = [round (value * 100, 4) for value in values_rbfk]

create_simple_bar_graph(x=x_rbfk, y=y_rbfk, value_labels=True, colour="Red", title="RBF SVM Kernel (Model D)", save_path="Metrics/Graphs/pngs/SVCRBFKernelMetrics.png", dpi=400)

# Random Forest

with open("ML/RandomForest/optimizedMetrics.json", 'r') as file:
    rf_data = json.load(file)

# removing training time from json

if training_time in rf_data :
    rf_training_time_data = rf_data.pop(training_time)

values_rf = list(rf_data.values())

x_rf = list(rf_data.keys())
y_rf = [round (value * 100, 4) for value in values_rf]

create_simple_bar_graph(x=x_rf, y=y_rf, value_labels=True, colour="Red", title="Random Forest Metrics (Model F)", save_path="Metrics/Graphs/pngs/rfMetrics.png", dpi=400)

# Random Forest Pre-Optimised

with open("ML/RandomForest/metrics.json", 'r') as file:
    pre_optimised_rf_data = json.load(file)

# removing training time from json

if training_time in pre_optimised_rf_data :
    pre_optimised_rf_training_time_data = pre_optimised_rf_data.pop(training_time)

pre_optimised_values_rf = list(pre_optimised_rf_data.values())

x_preop_rf = list(pre_optimised_rf_data.keys())
y_preop_rf = [round (value * 100, 4) for value in pre_optimised_values_rf]

create_simple_bar_graph(x=x_preop_rf, y=y_preop_rf, value_labels=True, colour="Red", title="Increased Forest No. Random Forest Metrics (Model E)", save_path="Metrics/Graphs/pngs/preOptimisedRfMetrics.png", dpi=400)

# DNN

with open("DL/DNN/metricsWithDropout.json", 'r') as file:
    dnn_data = json.load(file)

# removing training time from json

if training_time in dnn_data :
    dnn_training_time_data = dnn_data.pop(training_time)

values_dnn = list(dnn_data.values())

x_dnn = list(dnn_data.keys())
y_dnn = [round (value * 100, 4) for value in values_dnn]

create_simple_bar_graph(x=x_dnn, y=y_dnn, value_labels=True, colour="Red", title="DNN Metrics With Dropout and Early Stopping (Model H)", save_path="Metrics/Graphs/pngs/dnnMetrics.png", dpi=400)

# pre-optimised DNN

with open("DL/DNN/optimizedMetrics.json", 'r') as file:
    pre_optimised_dnn_data = json.load(file)

# removing training time from json

if training_time in pre_optimised_dnn_data :
    pre_optimised_dnn_training_time_data = pre_optimised_dnn_data.pop(training_time)

pre_optimised_values_dnn = list(pre_optimised_dnn_data.values())

x_preop_dnn = list(pre_optimised_dnn_data.keys())
y_preop_dnn = [round (value * 100, 4) for value in pre_optimised_values_dnn]

create_simple_bar_graph(x=x_preop_dnn, y=y_preop_dnn, value_labels=True, colour="Red", title="Pre-Optimised DNN Metrics (Model G)", save_path="Metrics/Graphs/pngs/preOptimisedDnnMetrics.png", dpi=400)

# dnn with early stopping, dropout, and extra layers

with open("DL/DNN/metricsWithDropoutSeveralLayers.json", 'r') as file:
    several_layers_dnn_data = json.load(file)

# removing training time from json

if training_time in several_layers_dnn_data :
    several_layers_dnn_training_time_data = several_layers_dnn_data.pop(training_time)

values_several_layers_dnn_data = list(several_layers_dnn_data.values())

x_svrl_dnn = list(several_layers_dnn_data.keys())
y_svrl_dnn = [round (value * 100, 4) for value in values_several_layers_dnn_data]

create_simple_bar_graph(x=x_svrl_dnn, y=y_svrl_dnn, value_labels=True, colour="Red", title="6 Layer DNN Model Metrics (Model I)", save_path="Metrics/Graphs/pngs/severalLayerDnnMetrics.png", dpi=400)

# CNN

with open("DL/CNN/optimizedMetricsWithDropout.json", 'r') as file:
    cnn_data = json.load(file)

# removing training time from json

if training_time in cnn_data :
    cnn_training_time_data = cnn_data.pop(training_time)

values_cnn = list(cnn_data.values())

x_cnn = list(cnn_data.keys())
y_cnn = [round (value * 100, 4) for value in values_cnn]

create_simple_bar_graph(x=x_cnn, y=y_cnn, value_labels=True, colour="Red", title="CNN With Dropout Metrics (Model L)", save_path="Metrics/Graphs/pngs/cnnMetricsWithDropout.png", dpi=400)

# pre optimised CNN

with open("DL/CNN/optimizedMetrics.json", 'r') as file:
    pre_optimised_cnn_data = json.load(file)

# removing training time from json

if training_time in pre_optimised_cnn_data :
    pre_optimised_cnn_training_time_data = pre_optimised_cnn_data.pop(training_time)

pre_optimised_values_cnn = list(pre_optimised_cnn_data.values())

x_preop_cnn = list(pre_optimised_cnn_data.keys())
y_preop_cnn = [round (value * 100, 4) for value in pre_optimised_values_cnn]

create_simple_bar_graph(x=x_preop_cnn, y=y_preop_cnn, value_labels=True, colour="Red", title="Pre-Optimised CNN Metrics (Model J)", save_path="Metrics/Graphs/pngs/preOptimisedCnnMetrics.png", dpi=400)

# CNN with early stopping

with open("DL/CNN/metricsWithEarlyStopping.json", 'r') as file:
    early_stopping_cnn_data = json.load(file)

# removing training time from json

if training_time in early_stopping_cnn_data :
    early_stopping_cnn_training_time_data = early_stopping_cnn_data.pop(training_time)

values_early_stopping_cnn_data = list(early_stopping_cnn_data.values())

x_es_cnn = list(early_stopping_cnn_data.keys())
y_es_cnn = [round (value * 100, 4) for value in values_early_stopping_cnn_data]

create_simple_bar_graph(x=x_es_cnn, y=y_es_cnn, value_labels=True, colour="Red", title="CNN With Early Stopping Metrics (Model K)", save_path="Metrics/Graphs/pngs/cnnWithEarlyStoppingMetrics.png", dpi=400)

# Training Times graph

training_times = {
    "A" : linear_svc_training_time_data,
    "B" : linear_kernel_training_time_data,
    "C" : poly_kernel_training_time_data,
    "D" : rbf_kernel_training_time_data,
    "E" : pre_optimised_rf_training_time_data,
    "F" : rf_training_time_data,
    "G" : pre_optimised_dnn_training_time_data,
    "H" : dnn_training_time_data,
    "I" : several_layers_dnn_training_time_data,
    "J" : pre_optimised_cnn_training_time_data,
    "K" : cnn_training_time_data,
    "L" : early_stopping_cnn_training_time_data
}

values_times = list(training_times.values())

x_times = list(training_times.keys())
y_times = [round(value, 1) for value in values_times]

create_simple_bar_graph(x=x_times, y=y_times, value_labels=True, colour="Blue", title="All Training Times (seconds)", save_path="Metrics/Graphs/pngs/allTrainingTimesMetrics.png", dpi=400)

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

# Fastest Training Times

fastest_training_times = {
    "SVM " : linear_svc_training_time_data,
    "Random Forest" : pre_optimised_rf_training_time_data,
    "DNN" : dnn_training_time_data,
    "CNN" : early_stopping_cnn_training_time_data
}

fastest_training_times_values = list(fastest_training_times.values())

x_fastest_times = list(pre_op_training_times.keys())
y_fastest_times = [round(value, 2) for value in fastest_training_times_values]

create_simple_bar_graph(x=x_fastest_times, y=y_fastest_times, value_labels=True, colour="Blue", title="Fastest Training Times (seconds)", save_path="Metrics/Graphs/pngs/fastestTrainingTimes.png", dpi=400)