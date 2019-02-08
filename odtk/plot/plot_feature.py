def plot_feature(dataset,
                 occupied_color="#009250",
                 unoccupied_color="#920000",
                 density_color="#00009250",
                 unit={},
                 file_name="output.png"):
    import matplotlib.pyplot as plt
    from numpy import concatenate

    dataset = dataset.copy()

    datas = concatenate((dataset.occupancy, dataset.data), axis=1)
    occupancy = dataset.occupancy.reshape((dataset.occupancy.shape[0],))
    occupied = datas[occupancy > 0, :]
    unoccupied = datas[occupancy < 0.5, :]

    header = ["Occupancy"] + dataset.feature_list
    all_unit = {"HumidityRatio": "(kg-w/kg-a)", "Humidity": "(%)", "CO2": "(ppm)", "Light": "(Lux)",
                "Temperature": "(Celsius)"}
    all_unit.update(unit)

    fig = plt.figure(figsize=(8, 8))  # Notice the equal aspect ratio
    ax = [fig.add_subplot(len(header), len(header), i * len(header) + j + 1)
          for i in range(len(header) - 1, -1, -1) for j in range(len(header))]

    for x in range(len(header)):
        for y in range(len(header)):
            current = ax[y * len(header) + x]
            y_min = datas[:, y].min()
            y_max = datas[:, y].max()

            if x != y:
                current.scatter(x=occupied[:, x],
                                y=occupied[:, y],
                                c=occupied_color,
                                s=1)
                current.scatter(x=unoccupied[:, x],
                                y=unoccupied[:, y],
                                c=unoccupied_color,
                                s=1)

            else:
                bin_value, _, _ = current.hist(datas[:, x], density=True, bins=40, color=density_color)
                y_min = 0
                y_max = max(bin_value)

            x_min = datas[:, x].min()
            x_max = datas[:, x].max()
            margin_ratio = 0.1
            current.set_xlim(x_min - (x_max - x_min) * margin_ratio,
                             x_max + (x_max - x_min) * margin_ratio)
            current.set_ylim(y_min - (y_max - y_min) * margin_ratio,
                             y_max + (y_max - y_min) * margin_ratio)
            if x:
                current.set_yticks([])
                current.set_yticklabels([])
            else:
                current.set_yticks([y_min, y_max])
                current.set_yticklabels(["%.1f" % datas[:, y].min(), "%.1f" % datas[:, y].max()],
                                        rotation=90, rotation_mode="anchor", ha="center")
                pad = 0
                if y % 2:
                    pad = 12
                    current.tick_params(axis='y', pad=3 + pad)
                current.set_ylabel(header[y] + "\n" + all_unit.get(header[y], ""), labelpad=20 - pad, weight="bold")

            if y:
                current.set_xticks([])
                current.set_xticklabels([])
            else:
                current.set_xticks([x_min, x_max])
                current.set_xticklabels(["%.1f" % x_min, "%.1f" % x_max])
                current.set_xlabel(header[x], labelpad=20)
                pad = 0
                if x % 2:
                    pad = 12
                    current.tick_params(axis='x', pad=3 + pad)
                current.set_xlabel(header[x] + "\n" + all_unit.get(header[x], ""), labelpad=20 - pad, weight="bold")

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(file_name, transparent=True, pad_inches=0)
    plt.show()
