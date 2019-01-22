def plot_feature(dataset, occupied_color="rgb(255, 0, 0, 1)", unoccupied_color="rgba(0, 255, 0, 0.5)"):
    import plotly as py
    import plotly.graph_objs as go
    import plotly.io as pio
    from numpy import concatenate

    dataset = dataset.copy()
    dataset.remove_feature([dataset.header_info[dataset.time_column], "id"])

    datas = concatenate((dataset.occupancy, dataset.data), axis=1)
    occupancy = dataset.occupancy.reshape((dataset.occupancy.shape[0],))
    occupied = datas[occupancy > 0, :]
    unoccupied = datas[occupancy < 0.5, :]

    data = list()
    header = ["Occupancy"] + dataset.header
    width = [i / len(header) for i in range(len(header) + 1)]

    for x in range(len(header)):
        for y in range(len(header)):
            if x == y:
                data.append(go.Histogram(x=datas[:, x],
                                         histnorm="probability",
                                         xaxis='x' + str(x * len(header) + y + 1),
                                         yaxis='y' + str(x * len(header) + y + 1),
                                         showlegend=False,
                                         # nbinsx=10,
                                         marker=dict(color="blue")))
            else:
                data.append(go.Scatter(x=occupied[:, x],
                                       y=occupied[:, y],
                                       mode="markers",
                                       xaxis='x' + str(x * len(header) + y + 1),
                                       yaxis='y' + str(x * len(header) + y + 1),
                                       showlegend=False,
                                       marker=dict(color=occupied_color,
                                                   size=3)))
                data.append(go.Scatter(x=unoccupied[:, x],
                                       y=unoccupied[:, y],
                                       mode="markers",
                                       xaxis='x' + str(x * len(header) + y + 1),
                                       yaxis='y' + str(x * len(header) + y + 1),
                                       showlegend=False,
                                       marker=dict(color=unoccupied_color,
                                                   size=3)))

    layout = dict()
    for x in range(len(header)):
        for y in range(len(header)):
            layout["xaxis" + str(x * len(header) + y + 1)] = dict(domain=[width[x], width[x + 1]],
                                                                  showgrid=False,
                                                                  mirror=True,
                                                                  showline=True,
                                                                  showticklabels=not bool(y),
                                                                  anchor='y' + str(x * len(header) + y + 1),
                                                                  nticks=3,
                                                                  tickangle=45,
                                                                  zeroline=False,
                                                                  title=header[x] if not bool(y) else None)
            layout["yaxis" + str(x * len(header) + y + 1)] = dict(domain=[width[y], width[y + 1]],
                                                                  showgrid=False,
                                                                  mirror=True,
                                                                  showline=True,
                                                                  showticklabels=not bool(x),
                                                                  anchor='x' + str(x * len(header) + y + 1),
                                                                  nticks=3,
                                                                  tickangle=45,
                                                                  zeroline=False,
                                                                  title=header[y] if not bool(x) else None)

    for x in range(len(header)):
        minimum = datas[:, x].min()
        maximum = datas[:, x].max()
        range_a = minimum - (maximum - minimum) * 0.1
        range_b = maximum + (maximum - minimum) * 0.1
        layout["xaxis" + str(x * len(header) + x + 1)]["range"] = [range_a, range_b]
    layout["yaxis1"]["showticklabels"] = False
    layout["yaxis" + str(len(header) + 1)]["showticklabels"] = True
    layout["yaxis" + str(len(header) + 1)]["anchor"] = "x1"

    fig = go.Figure(data=data, layout=go.Layout(layout))
    # py.offline.plot(fig, image="png", image_filename="fig", image_width=800, image_height=800)
    pio.write_image(fig, 'map1.png', width=800, height=800, validate=False)
