def plot_occupancy_swarm(datasets,
                         time_range=(0, 24 * 60 * 60),
                         weekly=False,
                         bin_size=60 * 10,
                         orientation="horizontal",
                         room_level=False,
                         file_name="output.png",
                         evaluation=False,
                         size=1.5):
    from time import mktime, gmtime
    from numpy import ix_, arange, histogram,  ndarray
    import matplotlib.pyplot as plt
    import seaborn as sns

    if not isinstance(datasets, dict):
        raise TypeError("Datasets must be a dictionary!")

    time_only = dict()
    i = 1

    for dataset in datasets:
        if not evaluation:
            if room_level:
                dataset = datasets[dataset]
                for room in dataset.room_list:
                    data, occupancy = dataset[room]
                    time_only[room] = data[ix_(occupancy.flatten() > 0, [dataset.time_column])].flatten()
            else:
                time_only[dataset] = datasets[dataset].data[ix_(datasets[dataset].occupancy.flatten() > 0,
                                                                [datasets[dataset].time_column])].flatten()
        else:
            time_only.update(datasets)

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

    if not isinstance(ax, ndarray):
        ax = [ax]

    for name in time_only:

        time_float = time_only[name] - mktime(gmtime(0))
        time_float %= (24 * 60 * 60 * ((weekly * 6) + 1))
        print(time_float)

        eval("sns.swarmplot(" + xy[0] + "=time_float, ax=ax[i - 1], size=size)")

        ax[i - 1].set_xticks([])
        ax[i - 1].set_xticklabels([])
        ax[i - 1].set_yticks([])
        ax[i - 1].set_yticklabels([])

        if orientation != "vertical" and i == 1 or orientation == "vertical" and i == len(time_only):
            eval("ax[i - 1].set_" + xy[0] + "ticks(arange(0, 24 * 60 * 60 + 61, 60 * 60 * 3))")
            eval("ax[i - 1].set_" + xy[0] + "ticklabels(time_label)")

        if orientation == "vertical":
            ax[i - 1].set_ylabel(name, rotation="horizontal", ha="right", labelpad=10)
            ax[i - 1].set_xlim((0, 24 * 60 * 60))
        else:
            ax[i - 1].set_xlabel(name)
            ax[i - 1].set_ylim((0, 24 * 60 * 60))
            ax[i - 1].invert_yaxis()

        ax[i - 1].set_frame_on(False)
        i += 1

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(file_name, transparent=True, pad_inches=0)
    plt.show()
