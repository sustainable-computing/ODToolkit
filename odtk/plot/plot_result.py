def plot_result(result, threshold=None, group_by=0,
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
