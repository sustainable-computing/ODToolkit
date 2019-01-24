def plot_one(result, threshold="None", group_by=0, dataset=None, model=None, metric=None, fixed="auto"):
    import matplotlib.pyplot as plt
    from numpy import arange

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

    valid = eval("result[2]" + threshold).sum(axis=0) == result[2].shape[0]
    legends = [legends[i] for i in range(len(legends)) if valid[i]]

    result[2] = result[2][:, valid]

    w = 1 / (len(legends) + 2)
    x = arange(len(x_labels), dtype=float)

    fig, ax = plt.subplots(figsize=(8, 6))
    hatches = ['/', '\\', '', '-', '+', 'x', 'o', 'O', '.', '*']

    for i in range(len(legends)):
        ax.bar(x, result[2][:, i], width=w, label=legends[i], hatch=hatches[i % len(hatches)])
        x += w

    plt.xticks(x - (len(legends) + 1) / 2 * w, x_labels)
    plt.legend()
    plt.show()

    # layout = go.Layout(
    #     barmode='group',
    # )
    #
    # fig = go.Figure(data=traces, layout=layout)
    # py.offline.plot(fig)

    return

    import plotly as py
    import plotly.graph_objs as go
    result = list(result.get_result(dataset=dataset, model=model, metric=metric))

    if group_by:
        x_labels = result[1]
        legends = result[0]
        result[2] = result[2].T
    else:
        x_labels = result[0]
        legends = result[1]

    if legend is None:
        valid = eval("result[2]" + threshold).sum(axis=0) == result[2].shape[0]
        legends = [legends[i] for i in range(len(legends)) if valid[i]]
    else:
        valid = [legends[i] in legend for i in range(len(legends))]
        legends = legend[:]

    result[2] = result[2][:, valid]

    traces = list()

    for i in range(len(legends)):
        traces.append(go.Bar(x=x_labels,
                             y=result[2][:, i],
                             name=legends[i]))

    layout = go.Layout(
        barmode='group',
    )

    fig = go.Figure(data=traces, layout=layout)
    py.offline.plot(fig)
