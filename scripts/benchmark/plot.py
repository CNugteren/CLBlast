# This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This file follows the
# PEP8 Python style guide and uses a max-width of 120 characters per line.
#
# Author(s):
#   Cedric Nugteren <www.cedricnugteren.nl>

import utils

import matplotlib
matplotlib.use('Agg')
from matplotlib import rcParams
import matplotlib.pyplot as plt

# Colors
BLUEISH = [c / 255.0 for c in [71, 101, 177]]  # #4765b1
REDISH = [c / 255.0 for c in [214, 117, 104]]  # #d67568
PURPLISH = [c / 255.0 for c in [85, 0, 119]]  # #550077
GREEN = [c / 255.0 for c in [144, 224, 98]] # #90e062
COLORS = [BLUEISH, REDISH, PURPLISH, GREEN]
MARKERS = ["o-", "x-", ".-"]


def plot_graphs(results, file_name, num_rows, num_cols,
                x_keys, y_keys, titles, x_labels, y_labels,
                label_names, title, tight_plot, verbose):
    assert len(results) == num_rows * num_cols
    assert len(results) != 1
    assert len(x_keys) == len(results)
    assert len(y_keys) == len(results)
    assert len(titles) == len(results)
    assert len(x_labels) == len(results)
    assert len(y_labels) == len(results)

    # Tight plot (for in a paper or presentation) or regular (for display on a screen)
    if tight_plot:
        plot_size = 5
        w_space = 0.20
        h_space = 0.39
        title_from_top = 0.11
        legend_from_top = 0.17
        legend_from_top_per_item = 0.04
        x_label_from_bottom = 0.09
        legend_spacing = 0.0
        font_size = 15
        font_size_legend = 13
        font_size_title = font_size
        bounding_box = "tight"
    else:
        plot_size = 8
        w_space = 0.15
        h_space = 0.22
        title_from_top = 0.09
        legend_from_top = 0.10
        legend_from_top_per_item = 0.07
        x_label_from_bottom = 0.06
        legend_spacing = 0.8
        font_size = 15
        font_size_legend = font_size
        font_size_title = 18
        bounding_box = None  # means not 'tight'

    # Initializes the plot
    size_x = plot_size * num_cols
    size_y = plot_size * num_rows
    rcParams.update({'font.size': font_size})
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(size_x, size_y), facecolor='w', edgecolor='k')
    fig.text(.5, 0.92, title, horizontalalignment="center", fontsize=font_size_title)
    plt.subplots_adjust(wspace=w_space, hspace=h_space)

    # Loops over each subplot
    for row in range(num_rows):
        for col in range(num_cols):
            index = row * num_cols + col
            result = results[index]
            ax = axes.flat[index]
            plt.sca(ax)
            print("[plot] Plotting subplot %d" % index)

            # Sets the x-axis labels
            x_list = [[r[x_key] for r in result] for x_key in x_keys[index]]
            x_ticks = [",".join([utils.float_to_kilo_mega(v) for v in values]) for values in zip(*x_list)]
            x_location = range(len(x_ticks))

            # Optional sparsifying of the labels on the x-axis
            if tight_plot and len(x_location) > 10:
                x_ticks = [v if not (i % 2) else "" for i, v in enumerate(x_ticks)]

            # Sets the y-data
            y_list = [[r[y_key] if y_key in r.keys() else 0 for r in result] for y_key in y_keys[index]]
            y_max = [max(y) if len(y) else 1 for y in y_list]
            y_max = max(y_max) if len(y_list) > 0 else 1

            # Sets the axes
            y_rounding = 10 if y_max < 80 else 50 if y_max < 400 else 200
            y_axis_limit = (y_max * 1.2) - ((y_max * 1.2) % y_rounding) + y_rounding
            plt.ylim(ymin=0, ymax=y_axis_limit)
            plt.xticks(x_location, x_ticks, rotation='vertical')

            # Sets the labels
            ax.set_title(titles[index], y=1.0 - title_from_top, fontsize=font_size)
            if col == 0 or y_labels[index] != y_labels[index - 1]:
                ax.set_ylabel(y_labels[index])
            ax.set_xlabel(x_labels[index])
            ax.xaxis.set_label_coords(0.5, x_label_from_bottom)

            # Plots the graph
            assert len(COLORS) >= len(y_keys[index])
            assert len(MARKERS) >= len(y_keys[index])
            assert len(label_names) == len(y_keys[index])
            for i in range(len(y_keys[index])):
                color = COLORS[i]
                marker = MARKERS[i]
                if label_names[i] in ["CLBlast", "CLBlast FP32"]:
                    color = BLUEISH
                    marker = "o-"
                elif label_names[i] in ["CLBlast FP16"]:
                    color = PURPLISH
                    marker = ".-"
                elif label_names[i] in ["clBLAS", "clBLAS FP32", "clBLAS (non-batched)"]:
                    color = REDISH
                    marker = "x-"
                elif label_names[i] in ["cuBLAS", "cuBLAS (non-batched)"]:
                    color = GREEN
                    marker = ".-"
                ax.plot(x_location, y_list[i], marker, label=label_names[i], color=color)

            # Sets the legend
            leg = ax.legend(loc=(0.02, 1.0 - legend_from_top - legend_from_top_per_item * len(y_keys[index])),
                            handletextpad=0.1, labelspacing=legend_spacing, fontsize=font_size_legend)
            leg.draw_frame(False)

    # Saves the plot to disk
    print("[benchmark] Saving plot to '" + file_name + "'")
    fig.savefig(file_name, bbox_inches=bounding_box)
