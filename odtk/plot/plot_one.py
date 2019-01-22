def plot_one(result, threshold="<= 1", group_by=0, dataset=None, model=None, metric=None):
    import plotly as py
    import plotly.graph_objs as go
    result = list(result.get_result(dataset=dataset, model=model, metric=metric))

    if dataset is None and model is None and metric is None:
        raise ValueError("You have to set one value")
    if group_by not in (0, 1):
        raise ValueError("group_by can only be 0 or 1")

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
