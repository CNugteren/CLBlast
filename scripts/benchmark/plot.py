# This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This file follows the
# PEP8 Python style guide and uses a max-width of 120 characters per line.
#
# Author(s):
#   Cedric Nugteren <www.cedricnugteren.nl>

import utils

from matplotlib import rcParams
import matplotlib.pyplot as plt

# Tight plot (for in a paper or presentation) or regular (for display on a screen)
TIGHT_PLOT = False
if TIGHT_PLOT:
    PLOT_SIZE = 5
    W_SPACE = 0.20
    H_SPACE = 0.39
    TITLE_FROM_TOP = 0.11
    LEGEND_FROM_TOP = 0.17
    LEGEND_FROM_TOP_PER_ITEM = 0.04
    X_LABEL_FROM_BOTTOM = 0.09
    LEGEND_SPACING = 0.0
    FONT_SIZE = 15
    FONT_SIZE_LEGEND = 13
    FONT_SIZE_TITLE = FONT_SIZE
else:
    PLOT_SIZE = 8
    W_SPACE = 0.15
    H_SPACE = 0.22
    TITLE_FROM_TOP = 0.09
    LEGEND_FROM_TOP = 0.10
    LEGEND_FROM_TOP_PER_ITEM = 0.07
    X_LABEL_FROM_BOTTOM = 0.06
    LEGEND_SPACING = 0.8
    FONT_SIZE = 15
    FONT_SIZE_LEGEND = FONT_SIZE
    FONT_SIZE_TITLE = 18

# Colors
BLUEISH = [c / 255.0 for c in [71, 101, 177]]  # #4765b1
REDISH = [c / 255.0 for c in [214, 117, 104]]  # #d67568
PURPLISH = [c / 255.0 for c in [85, 0, 119]]  # #550077
COLORS = [BLUEISH, REDISH, PURPLISH]
MARKERS = ["o-", "x-", ".-"]


def plot_graphs(results, file_name, num_rows, num_cols,
                x_keys, y_keys, titles, x_labels, y_labels,
                label_names, title, verbose):
    assert len(results) == num_rows * num_cols
    assert len(results) != 1
    assert len(x_keys) == len(results)
    assert len(y_keys) == len(results)
    assert len(titles) == len(results)
    assert len(x_labels) == len(results)
    assert len(y_labels) == len(results)

    # Initializes the plot
    size_x = PLOT_SIZE * num_cols
    size_y = PLOT_SIZE * num_rows
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(size_x, size_y), facecolor='w', edgecolor='k')
    fig.text(.5, 0.92, title, horizontalalignment="center", fontsize=FONT_SIZE_TITLE)
    plt.subplots_adjust(wspace=W_SPACE, hspace=H_SPACE)
    rcParams.update({'font.size': FONT_SIZE})

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
            if TIGHT_PLOT and len(x_location) > 10:
                x_ticks = [v if not (i % 2) else "" for i, v in enumerate(x_ticks)]

            # Sets the y-data
            y_list = [[r[y_key] for r in result] for y_key in y_keys[index]]
            y_max = max([max(y) for y in y_list])

            # Sets the axes
            y_rounding = 10 if y_max < 80 else 50 if y_max < 400 else 200
            y_axis_limit = (y_max * 1.2) - ((y_max * 1.2) % y_rounding) + y_rounding
            plt.ylim(ymin=0, ymax=y_axis_limit)
            plt.xticks(x_location, x_ticks, rotation='vertical')

            # Sets the labels
            ax.set_title(titles[index], y=1.0 - TITLE_FROM_TOP)
            if col == 0 or y_labels[index] != y_labels[index - 1]:
                ax.set_ylabel(y_labels[index])
            ax.set_xlabel(x_labels[index])
            ax.xaxis.set_label_coords(0.5, X_LABEL_FROM_BOTTOM)

            # Plots the graph
            assert len(COLORS) >= len(y_keys[index])
            assert len(MARKERS) >= len(y_keys[index])
            assert len(label_names) == len(y_keys[index])
            for i in range(len(y_keys[index])):
                ax.plot(x_location, y_list[i], MARKERS[i], label=label_names[i], color=COLORS[i])

            # Sets the legend
            leg = ax.legend(loc=(0.02, 1.0 - LEGEND_FROM_TOP - LEGEND_FROM_TOP_PER_ITEM * len(y_keys[index])),
                            handletextpad=0.1, labelspacing=LEGEND_SPACING, fontsize=FONT_SIZE_LEGEND)
            leg.draw_frame(False)

    # Saves the plot to disk
    fig.savefig(file_name, bbox_inches='tight')
    plt.show()
