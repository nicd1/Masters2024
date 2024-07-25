import matplotlib.pyplot as plt
import numpy as np

def create_grouped_bar_graph(data, bar_labels, colours, title, x_label, y_label, save_path, dpi, legend_labels=None):
    plt.clf()
    num_bars = len(data)
    width = 0.2
    x = np.arange(len(bar_labels))

    for i in range(num_bars):
        plt.bar(x + i * width, data[i], width, color=colours[i])

    plt.xticks(x + (num_bars - 1) * width / 2, bar_labels)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if legend_labels:
        plt.legend(legend_labels, loc='upper center', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)

    plt.savefig(save_path, dpi=dpi)
    plt.close()

def create_simple_bar_graph(x, y, colour, title, save_path, dpi, value_labels=False, x_label=None, y_label=None):
    plt.clf()
    plt.bar(x, y, color=colour)
    plt.title(title)

    if x_label:
        plt.xlabel(x_label)

    if y_label:
        plt.ylabel(y_label)

    if value_labels:
        for i in range(len(x)):
            plt.text(i, y[i], y[i], ha = 'center')

    plt.savefig(save_path, dpi=dpi)
    plt.close()


