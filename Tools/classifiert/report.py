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

    plot_statistic(statistics, f"Wall Clock Time (Train) :: {num_threads} thread(s)", "Time (s)", key="train-wall-clock-time")
    plot_statistic(statistics, f"CPU Time (Train) :: {num_threads} thread(s)", "Time (s)",
                   getter_func=lambda d, _: d["train-user-time"] + d["train-system-time"])
    plot_statistic(statistics, f"Percent CPU (Train) :: {num_threads} thread(s)", "%", key="train-percent-cpu")
    plot_statistic(statistics, f"Maximum RSS (Train) :: {num_threads} thread(s)", "RSS (GB)",
                   getter_func=lambda d, _: d["train-max-rss"] / 1_000_000)
    plot_statistic(statistics, f"Maximum node count", "No. of nodes", key="train-max-node-count")
    plot_statistic(statistics, f"Maximum tree depth", "Levels", key="train-max-tree-depth")

    plot_statistic(statistics, f"Wall Clock Time (Train) :: {num_threads} thread(s)", "Time (s)", key="test-wall-clock-time")
    plot_statistic(statistics, f"CPU Time (Test) :: {num_threads} thread(s)", "Time (s)",
                   getter_func=lambda d, _: d["test-user-time"] + d["test-system-time"])
    plot_statistic(statistics, f"Percent CPU (Test) :: {num_threads} thread(s)", "%", key="test-percent-cpu")
    plot_statistic(statistics, f"Maximum RSS (Test) :: {num_threads} thread(s)", "RSS (GB)",
                   getter_func=lambda d, _: d["test-max-rss"] / 1_000_000)
    plot_statistic(statistics, f"Model Load Wall Clock Time", "Time (s)", key="test-model-load-time")
    plot_statistic(statistics, f"Data Load Wall Clock Time", "Time (s)", key="test-data-load-time")
    plot_statistic(statistics, f"Classification Wall Clock Time", "Time (s)", key="test-classification-time")
    plot_statistic(statistics, f"Label Store Wall Clock Time", "Time (s)", key="test-label-store-time")
    plot_statistic(statistics, f"Accuracy", "Accuracy", key="test-accuracy")
    plot_statistic(statistics, f"P4-metric", "P4-metric", key="test-P4-metric")
    plot_statistic(statistics, f"Positive Predictive Value", "PPV", key="test-ppv")
    plot_statistic(statistics, f"True Positive Rate", "True Positive Rate", key="test-tpr")
    plot_statistic(statistics, f"True Negative Rate", "True Negative Rate", key="test-tnr")
    plot_statistic(statistics, f"Negative Predictive Value", "NPV", key="test-npv")

    from matplotlib.backends.backend_pdf import PdfPages
    report = PdfPages(report_filename)
    for fignum in plt.get_fignums():
        figure = plt.figure(fignum)
        figure.savefig(report, format="pdf")
        plt.close(fignum)
    report.close()
