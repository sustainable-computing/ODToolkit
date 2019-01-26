def plot_occupancy_perc(datasets,
                        time_range=(0, 24 * 60 * 60),
                        weekly=False,
                        bin_size=60 * 10,
                        orientation="horizontal",
                        room_level=False):
    from time import mktime, gmtime
    from numpy import ix_, arange
    import matplotlib.pyplot as plt
    from pprint import pprint

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
        fig, ax = plt.subplots(nrows=len(time_only), sharex=True)
    else:
        fig, ax = plt.subplots(ncols=len(time_only), sharey=True)

    for name in time_only:

        time_float = time_only[name] - mktime(gmtime(0))
        time_float %= (24 * 60 * 60 * ((weekly * 6) + 1))
        # Wait to finish weekly

        xy_order = "yx"
        if orientation == "vertical":
            xy_order = "xy"

        ax[i - 1].hist(time_float,
                       arange(time_range[0], time_range[1] + 1, bin_size),
                       orientation=orientation,
                       density=True)

        ax[i - 1].set_xticks([])
        ax[i - 1].set_xticklabels([])
        ax[i - 1].set_yticks([])
        ax[i - 1].set_yticklabels([])

        if orientation != "vertical" and i == 1 or orientation == "vertical" and i == len(time_only):
            time_label = ["0 am"] + [str(i % 12 + 1) + [" am", " pm"][i // 12] for i in range(2, 24, 3)]
            eval("ax[i - 1].set_" + xy_order[0] + "ticks(arange(0, 24 * 60 * 60 + 61, 60 * 60 * 3))")
            eval("ax[i - 1].set_" + xy_order[0] + "ticklabels(time_label)")

        eval("ax[i - 1]." + xy_order[1] + "label(name)")
        ax[i - 1].set_frame_on(False)
        i += 1

    # plt.subplots_adjust(wspace=0, hspace=0.5)
    plt.show()
