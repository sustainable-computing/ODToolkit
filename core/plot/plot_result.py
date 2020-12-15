#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def plot_result(result,
                threshold=None,
                group_by=0,
                dataset=None,
                model=None,
                metric=None,
                fixed_category="auto",
                x_label="",
                y_label="",
                y_range=None,
                add_label=True,
                font_size=12,
                file_name=None,
                bar_size=2,
                add_line=False,
                **kwargs):
    """
    Plot the 2D result bar plot for the experimental result. The dimension with only one selected name
    is the dimension to flat

    :parameter result: the 3D result from the experiments
    :type result: core.evaluation.superclass.Result

    :parameter threshold: maximum score show in this figure
    :type threshold: float

    :parameter group_by: indicate using which dimension as the x-axis. Selection of ``0`` and ``1``
    :type group_by: int

    :parameter dataset: select the name of datasets want to present in this figure
    :type dataset: str or list(str)

    :parameter model: select the name of models want to present in this figure
    :type model: str or list(str)

    :parameter metric: select the name of metrics want to present in this figure
    :type metric: str or list(str)

    :parameter fixed_category: find which asix only have one value in order to create 2D result. If ``'auto'`` then
                               it will automatically find the dimension with only one value. Value must be ``'auto'``,
                               ``'dataset'``, ``'model'``, or ``'metric'``
    :type fixed_category: str

    :parameter y_range: the range of the y-axis. If ``None``, then use best fit range
    :type y_range: ``None`` or list(float)

    :parameter x_label: text label on x_axis
    :type x_label: str

    :parameter y_label: text label on y_axis
    :type y_label: str

    :parameter add_label: decide whether add labels to the x-axis
    :type add_label: bool

    :parameter font_size: the font size for all elements in the figure
    :type add_label: float

    :parameter bar_size: the width of the bar
    :type bar_size: float

    :parameter add_line: decide whether add trend line on histogram
    :type add_line: bool

    :parameter file_name: the file name of function's figure.
                          if None, then do not write figure to a file.
                          Otherwise, write figure to file_name
    :type file_name: str

    :parameter \*\*kwarg: other arguments for the matplotlib.pyplot.bar

    :return: None
    """
    import matplotlib.pyplot as plt
    from numpy import arange

    plt.rcParams.update({'font.size': font_size})

    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1.01 * height,
                    '%.2f' % height,
                    ha='center', va='bottom')

    if dataset is None and model is None and metric is None:
        raise ValueError("You have to set one value")
    if group_by not in (0, 1):
        raise ValueError("group_by can only be 0 or 1")

    result = list(result.get_result(dataset=dataset, model=model, metric=metric, fixed=fixed_category))

    if group_by:
        x_labels = result[1]
        legends = result[0]
        result[2] = result[2].T
    else:
        x_labels = result[0]
        legends = result[1]

    if threshold is None:
        valid = [True] * len(legends)
    else:
        valid = eval("result[2]" + threshold).sum(axis=0) == result[2].shape[0]
    legends = [legends[i] for i in range(len(legends)) if valid[i]]

    result[2] = result[2][:, valid]

    w = 1 / (len(legends) + bar_size)
    x = arange(len(x_labels), dtype=float)

    fig, ax = plt.subplots(figsize=(12, 8))
    hatches = ['/', '\\', '', '-', '+', 'x', 'o', 'O', '.', '*']
    markers = ['o', 'D', '+', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', '*', 'h', 'H', '.',
               'X', 'x', ',', 'd', '|', '_']

    for i in range(len(legends)):

        if add_line:
            line = ax.plot(result[2][:, i], label=legends[i],
                           marker=markers[i% len(markers)],
                           ms=15,
                           markeredgecolor="#000000ff",
                           markerfacecolor="#00000000")
        else:
            bars = ax.bar(x, result[2][:, i], width=w, label=legends[i], hatch=hatches[i % len(hatches)], **kwargs)
            if add_label:
                autolabel(bars)
            x += w

    plt.xlabel(x_label, fontweight='bold')
    plt.ylabel(y_label, fontweight='bold')
    if y_range is not None:
        plt.ylim(y_range)
    if add_line:
        plt.xticks(x, x_labels)
    else:
        plt.xticks(x - (len(legends) + 1) / 2 * w, x_labels)
    plt.legend()
    if file_name is not None:
        plt.savefig(file_name, transparent=True, pad_inches=0)
    plt.show()
