import matplotlib.pyplot as plt


def plot_statistic(statistics, title, y_axis_label, *, key=None, getter_func=dict.get):

    plt.figure()
    plt.title(title)
    plt.xlabel("No. of samples")
    plt.ylabel(y_axis_label)
    for classifier_name, classifier_statistics in statistics.items():
        x_values = [run_statistics["data_size"] for run_statistics in classifier_statistics]
        y_values = [getter_func(run_statistics, key) for run_statistics in classifier_statistics]
        plt.plot(x_values, y_values, linestyle="-", marker=".", label=classifier_name)
    plt.legend()


def write_report(report_filename, num_threads, statistics):

    plot_statistic(statistics, f"CPU Time :: {num_threads} thread(s)", "CPU Time (s)",
                   getter_func=lambda d, _: d["user_time"] + d["system_time"])
    plot_statistic(statistics, f"Maximum RSS :: {num_threads} thread(s)", "Maximum RSS (GB)",
                   getter_func=lambda d, _: d["max_rss"] / 1_000_000)
    plot_statistic(statistics, f"Accuracy", "Accuracy (%)", key="accuracy")
    plot_statistic(statistics, f"Maximum node count", "No. of nodes", key="node_count")
    plot_statistic(statistics, f"Maximum tree depth", "Levels", key="depth")

    from matplotlib.backends.backend_pdf import PdfPages
    report = PdfPages(report_filename)
    for fignum in plt.get_fignums():
        figure = plt.figure(fignum)
        figure.savefig(report, format="pdf")
        plt.close(fignum)
    report.close()
