#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def plot_occupancy_distribution(datasets,
                                bin_size=60 * 10,
                                orientation="horizontal",
                                room_level=False,
                                file_name=None,
                                skip_calculation=False,
                                size=1.5,
                                x_label="",
                                y_label="",
                                swarm=False):
    """
    Plot the distribution of when the room is occupied over a day (averaged) using histogram or swarmplot.
    Each Dataset in datasets have one distribution plot, and are shown in the same figure

    :parameter datasets: a set of Dataset corresponds to its name
    :type datasets: dict(str, core.data.dataset.Dataset)

    :parameter bin_size: number of seconds for each bin
    :type bin_size: int

    :parameter orientation: the direction or the plot. Selection of ``'vertical'`` or ``'horizontal'``
    :type orientation: str

    :parameter room_level: decide the result is separate for each room in each Dataset or
                           combine each dataset together
    :type room_level: bool

    :parameter skip_calculation: if datasets is a set of list contains only times,
                                 then function can skip calculation
    :type skip_calculation: bool

    :parameter size: the size of the dot in swarmplot
    :type size: float

    :parameter x_label: text label on x_axis
    :type x_label: str

    :parameter y_label: text label on y_axis
    :type y_label: str

    :parameter swarm: decide whether use histogram or swarmplot
    :type swarm: bool

    :parameter file_name: the file name of function's figure.
                          if None, then do not write figure to a file.
                          Otherwise, write figure to file_name
    :type file_name: str

    :return: None
    """
    from time import mktime, gmtime
    from numpy import ix_, arange, histogram, ndarray
    import matplotlib.pyplot as plt
    from seaborn import swarmplot

    if not isinstance(datasets, dict):
        raise TypeError("Datasets must be a dictionary!")

    time_only = dict()
    i = 1

    for dataset in datasets:
        if not skip_calculation:
            if room_level:
                dataset = datasets[dataset]
                for room in dataset.room_list:
                    data, occupancy = dataset[room]
                    time_only[room] = data[ix_(occupancy.flatten() > 0, [dataset.time_column])].flatten()
            else:
                time_only[dataset] = datasets[dataset].data[ix_(datasets[dataset].occupancy.flatten() > 0,
                                                                [datasets[dataset].time_column_index])].flatten()
        else:
            time_only.update(datasets)

    if orientation == "vertical":
        xy = "xy"
        fig, ax = plt.subplots(nrows=len(time_only))
    else:
        xy = "yx"
        fig, ax = plt.subplots(ncols=len(time_only))

    if not isinstance(ax, ndarray):
        ax = [ax]

    ax_all = fig.add_subplot(111, zorder=-1)
    time_label = ['12 am', '3 am', '6 am', '9 am', '12 pm', '3 pm', '6 pm', '9 pm', '12 am']
    eval("ax_all.set_" + xy[0] + "ticks(arange(0, 24 * 60 * 60 + 61, 60 * 60 * 3))")
    ax_all.tick_params(left=False, right=False, bottom=False, top=False,
                       labelbottom=False, labelleft=False)
    eval("ax_all.get_shared_" + xy[0] + "_axes().join(ax_all, ax[0])")
    ax_all.grid(axis=xy[0], alpha=0.5)
    time_range = (0, 24 * 60 * 60)

    for name in time_only:

        time_float = time_only[name] - mktime(gmtime(0))
        time_float %= time_range[1]
        # Wait to finish weekly

        # n, bins, _ = ax[i - 1].hist(time_float,
        #                             arange(time_range[0], time_range[1] + 1, bin_size),
        #                             orientation=orientation,
        #                             density=True)

        if swarm:
            eval("swarmplot(" + xy[0] + "=time_float, ax=ax[i - 1], size=size)")
        else:
            n, bins = histogram(time_float,
                                bins=arange(time_range[0], time_range[1] + 1, bin_size),
                                density=True)
            bins = bins[:-1]
            if orientation == "vertical":
                ax[i - 1].fill_between(bins, 0, n, alpha=0.5)
                ax[i - 1].plot(bins, n)
            else:
                ax[i - 1].fill_betweenx(bins, 0, n, alpha=0.5)
                ax[i - 1].plot(n, bins)

        ax[i - 1].set_xticks([])
        ax[i - 1].set_xticklabels([])
        ax[i - 1].set_yticks([])
        ax[i - 1].set_yticklabels([])

        if orientation != "vertical" and i == 1 or orientation == "vertical" and i == len(time_only):
            eval("ax[i - 1].set_" + xy[0] + "ticks(arange(0, 24 * 60 * 60 + 61, 60 * 60 * 3))")
            eval("ax[i - 1].set_" + xy[0] + "ticklabels(time_label)")

        if orientation == "vertical":
            ax[i - 1].set_ylabel(name, rotation="horizontal", ha="right", va="center", labelpad=10)
            ax[i - 1].set_xlim((0, 24 * 60 * 60))
            if not swarm:
                ax[i - 1].set_ylim((n.min() - (n.max() - n.min()) * 0.1,
                                    n.max() + (n.max() - n.min()) * 0.1))
        else:
            ax[i - 1].set_xlabel(name)
            ax[i - 1].set_ylim((0, 24 * 60 * 60))
            ax[i - 1].invert_yaxis()
            if not swarm:
                ax[i - 1].set_xlim((n.min() - (n.max() - n.min()) * 0.1,
                                    n.max() + (n.max() - n.min()) * 0.1))

        ax[i - 1].set_frame_on(False)
        i += 1

    plt.xlabel(x_label, fontweight='bold')
    plt.ylabel(y_label, fontweight='bold')
    plt.subplots_adjust(wspace=0, hspace=0)
    if file_name is not None:
        plt.savefig(file_name, transparent=True, pad_inches=0)
    plt.show()
