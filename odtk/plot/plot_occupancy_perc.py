def plot_occupancy_perc(datasets,
                        time_range=(0, 24 * 60 * 60),
                        weekly=False,
                        bin_size=60 * 10,
                        orientation="horizontal",
                        room_level=False):
    from time import mktime, gmtime
    from numpy import ix_, arange, histogram,concatenate
    import matplotlib.pyplot as plt

    if not isinstance(datasets, dict):
        raise TypeError("Datasets must be a dictionary!")

    time_only = dict()
    i = 1

    for dataset in datasets:
        if room_level:
            dataset = datasets[dataset]
            for room in dataset.room:
                data, occupancy = dataset[room]
                time_only[room] = data[ix_(occupancy.flatten() > 0, [dataset.time_column])].flatten()
        else:
            time_only[dataset] = datasets[dataset].data[ix_(datasets[dataset].occupancy.flatten() > 0,
                                                            [datasets[dataset].time_column])].flatten()

    if orientation == "vertical":
        xy = "xy"
        fig, ax = plt.subplots(nrows=len(time_only))
    else:
        xy = "yx"
        fig, ax = plt.subplots(ncols=len(time_only))

    ax_all = fig.add_subplot(111, zorder=-1)
    time_label = ['12 am', '3 am', '6 am', '9 am', '12 pm', '3 pm', '6 pm', '9 pm', '12 am']
    eval("ax_all.set_" + xy[0] + "ticks(arange(0, 24 * 60 * 60 + 61, 60 * 60 * 3))")
    ax_all.tick_params(left=False, right=False, bottom=False, top=False,
                       labelbottom=False, labelleft=False)
    eval("ax_all.get_shared_" + xy[0] + "_axes().join(ax_all, ax[0])")
    ax_all.grid(axis=xy[0], alpha=0.5)

    for name in time_only:

        time_float = time_only[name] - mktime(gmtime(0))
        time_float %= (24 * 60 * 60 * ((weekly * 6) + 1))
        # Wait to finish weekly

        # n, bins, _ = ax[i - 1].hist(time_float,
        #                             arange(time_range[0], time_range[1] + 1, bin_size),
        #                             orientation=orientation,
        #                             density=True)

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
            ax[i - 1].set_ylabel(name, rotation=45, ha="center", labelpad=20)
        else:
            ax[i - 1].set_xlabel(name)
        if orientation == "horizontal":
            ax[i - 1].invert_yaxis()
            ax[i - 1].set_xlim((n.min() - (n.max() - n.min()) * 0.1,
                                n.max() + (n.max() - n.min()) * 0.1))
        ax[i - 1].set_frame_on(False)
        i += 1

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
