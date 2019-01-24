def plot_one(result, threshold=None, group_by=0,
             dataset=None,
             model=None,
             metric=None,
             fixed="auto",
             x_label="",
             y_label="",
             y_range=None,
             line=[],
             **kwargs):
    import matplotlib.pyplot as plt
    from numpy import arange

    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                    '%.2f' % height,
                    ha='center', va='bottom')

    if dataset is None and model is None and metric is None:
        raise ValueError("You have to set one value")
    if group_by not in (0, 1):
        raise ValueError("group_by can only be 0 or 1")

    result = list(result.get_result(dataset=dataset, model=model, metric=metric, fixed=fixed))

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

    w = 1 / (len(legends) + 2)
    x = arange(len(x_labels), dtype=float)

    fig, ax = plt.subplots(figsize=(8, 6))
    hatches = ['/', '\\', '', '-', '+', 'x', 'o', 'O', '.', '*']

    for i in range(len(legends)):

        bars = ax.bar(x, result[2][:, i], width=w, label=legends[i], hatch=hatches[i % len(hatches)], **kwargs)
        autolabel(bars)
        if legends[i] in line:
            ax.plot(x, result[2][:, i])
        x += w

    plt.xlabel(x_label, fontweight='bold')
    plt.ylabel(y_label, fontweight='bold')
    if y_range is not None:
        plt.ylim(y_range)
    plt.xticks(x - (len(legends) + 1) / 2 * w, x_labels)
    plt.legend()
    plt.show()
