import plotly as py
import plotly.graph_objs as go


def plot_one_model(result, model, threshold="<= 1", group_by="dataset"):

    group = list(result.keys())
    legend = dict()
    for name in names:
        for metric in result[name][model].keys():
            legend[metric] = legend.get(metric, []) + [result[name][model][metric]]

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
