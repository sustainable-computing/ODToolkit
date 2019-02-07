def plot_occupancy(datasets, binary=True, total=True):
    import plotly as py
    import plotly.graph_objs as go
    from odtk.stats.occupancy_evaluation import occupancy_distribution_evaluation

    if not isinstance(datasets, dict):
        raise TypeError("Datasets must be a dictionary")

    final = dict()
    if total:
        for name in datasets.keys():
            final[name] = occupancy_distribution_evaluation(datasets[name], dataset_level=total)
    else:

        for name in datasets.keys():
            temp = occupancy_distribution_evaluation(datasets[name], dataset_level=total)
            for subdata in temp.keys():
                final[name + '-' + subdata] = temp[subdata]

    x_label = list(final.keys())

    if binary:
        y_occupied = [final[name]['occupied'][1] for name in x_label]
        y_unoccupied = [(1 - occupied) for occupied in y_occupied]

        trace1 = go.Bar(x=x_label,
                        y=y_occupied,
                        name="Occupied")
        trace2 = go.Bar(x=x_label,
                        y=y_unoccupied,
                        name="Un-occupied")

        data = [trace1, trace2]
        layout = go.Layout(barmode="stack")
        fig = go.Figure(data=data, layout=layout)
        py.offline.plot(fig)

    else:
        data = list()
        annotations = list()
        width = 1 / len(x_label)

        for i in range(len(x_label)):
            label = x_label[i]
            labels = list(final[label].keys())
            labels.remove('occupied')
            labels = list(map(int, labels))
            values = [final[label][occupancy][1] for occupancy in labels]
            pull = [0.2 if not occupancy else 0 for occupancy in labels]
            data.append(go.Pie(labels=labels,
                               values=values,
                               pull=pull,
                               textinfo="label+percent",
                               domain={'x': [i * width, (i + 1) * width],
                                       'y': [0, 1]},
                               name=label,
                               textposition="inside"))
            annotations.append({"text": label,
                                'x': (i + 0.5) * width,
                                'y': 0.9})

        fig = go.Figure(data=data, layout=go.Layout(annotations=annotations))

        py.offline.plot(fig)
