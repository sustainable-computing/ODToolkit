import plotly as py
import plotly.graph_objs as go


def plot_one_metric(result, metric, threshold="<= 1"):
    names = list(result.keys())
    metrics = dict()
    for name in names:
        for metric in result[name][metric].keys():
            metrics[metric] = metrics.get(metric, []) + [result[name][metric][metric]]

    valid = list()
    for metric in metrics.keys():
        if eval("max(metrics[metric])" + threshold):
            valid.append(metric)

    data = list()
    for metric in valid:
        data.append(go.Bar(x=names,
                           y=metrics[metric],
                           name=metric))

    layout = go.Layout(
        xaxis=dict(tickangle=-45),
        barmode='group',
    )

    fig = go.Figure(data=data, layout=layout)
    py.offline.plot(fig)
