#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def plot_start_end_distribution(start,
                                end,
                                bin_size=60 * 10,
                                orientation="horizontal",
                                file_name="output.png",
                                size=1.5,
                                x_label="",
                                y_label="",
                                swarm=False):
    """
    Plot the distribution of when the room becomes occupied at the first time over a day (averaged)
    and the distribution of when the room starts unoccupied at the last time over a day (averaged)
    using histogram or swarmplot. Each Dataset in datasets have one distribution plot,
    and are shown in the same figure

    :parameter start: a set of list contains only the time of first occupied in seconds
    :type start: list(int)

    :parameter end: a set of list contains only the time of last unoccupied in seconds
    :type end: list(int)

    :parameter orientation: the direction or the plot. Selection of ``'vertical'`` or ``'horizontal'``
    :type orientation: str

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
    from numpy import ix_, arange, histogram, ndarray
    import matplotlib.pyplot as plt
    from seaborn import swarmplot

    if not isinstance(start, dict) or not isinstance(end, dict):
        raise TypeError("Datasets must be a dictionary!")
    if len(start) != len(end):
        raise TypeError("Datasets must be a dictionary!")

    if orientation == "vertical":
        xy = "xy"
        fig, ax = plt.subplots(nrows=len(start) + 1)
        i = 2
        ax[0].set_xticks([])
        ax[0].set_xticklabels([])
        ax[0].set_yticks([])
        ax[0].set_yticklabels([])
        ax[0].set_xlim((0, 24 * 60 * 60))

        ax[0].set_frame_on(False)
    else:
        xy = "yx"
        fig, ax = plt.subplots(ncols=len(start) + 1)
        i = 1
        ax[-1].set_xticks([])
        ax[-1].set_xticklabels([])
        ax[-1].set_yticks([])
        ax[-1].set_yticklabels([])
        ax[-1].set_xlim((0, 24 * 60 * 60))

        ax[-1].set_frame_on(False)

    if not isinstance(ax, ndarray):
        ax = [ax]

    ax_all = fig.add_subplot(111, zorder=-1)
    time_label = ['12 am', '3 am', '6 am', '9 am', '12 pm', '3 pm', '6 pm', '9 pm', '12 am']
    eval("ax_all.set_" + xy[0] + "ticks(arange(0, 24 * 60 * 60 + 61, 60 * 60 * 3))")
    ax_all.tick_params(left=False, right=False, bottom=False, top=False,
                       labelbottom=False, labelleft=False)
    eval("ax_all.get_shared_" + xy[0] + "_axes().join(ax_all, ax[0])")
    ax_all.grid(axis=xy[0], alpha=0.5)



    for name in start:
        # Wait to finish weekly

        # n, bins, _ = ax[i - 1].hist(time_float,
        #                             arange(time_range[0], time_range[1] + 1, bin_size),
        #                             orientation=orientation,
        #                             density=True)

        if swarm:
            eval("swarmplot(" + xy[0] + "=time_float, ax=ax[i - 1], size=size)")
        else:
            n, bins = histogram(start[name],
                                bins=arange(0, 24 * 60 * 60 + 1, bin_size),
                                density=False)
            n2, bins2 = histogram(end[name],
                                  bins=arange(0, 24 * 60 * 60 + 1, bin_size),
                                  density=False)
            bins = bins[:-1]
            bins2 = bins2[:-1]
            if orientation == "vertical":
                ax[i - 1].fill_between(bins, 0, n, alpha=0.5,
                                       label="Occupancy start times")
                ax[i - 1].plot(bins, n)
                ax[i - 1].fill_between(bins2, 0, n2, alpha=0.5, hatch='.' * 6,
                                       label="Occupancy end times")
                ax[i - 1].plot(bins2, n2)
            else:
                ax[i - 1].fill_betweenx(bins, 0, n, alpha=0.5,
                                        label="Occupancy start times")
                ax[i - 1].plot(n, bins)
                ax[i - 1].fill_betweenx(bins2, 0, n2, alpha=0.5,
                                        label="Occupancy end times")
                ax[i - 1].plot(n2, bins2)

        ax[i - 1].set_xticks([])
        ax[i - 1].set_xticklabels([])
        ax[i - 1].set_yticks([])
        ax[i - 1].set_yticklabels([])

        if orientation != "vertical" and i == 1 or orientation == "vertical" and i == len(start) + 1:
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

    handles, labels = ax[1].get_legend_handles_labels()
    if orientation == "vertical":
        ax[0].legend(handles, labels, loc="center", ncol=2)
    else:
        ax[-1].legend(handles, labels, loc="center")
    # plt.xlabel(x_label, fontweight='bold')
    # plt.ylabel(y_label, fontweight='bold')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(file_name, transparent=True, pad_inches=0)
    plt.show()
