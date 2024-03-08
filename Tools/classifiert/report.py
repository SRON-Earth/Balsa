import matplotlib.pyplot as plt

def write_statistic(report, statistics, title, y_axis_label, *, key=None, getter_func=dict.get):

    plt.figure()
    plt.title(title)
    plt.xlabel("No. of samples")
    plt.ylabel(y_axis_label)
    plot_empty = True
    for classifier_name, classifier_statistics in statistics.items():
        x_values = [run_statistics["data_size"] for run_statistics in classifier_statistics]
        y_values = [getter_func(run_statistics, key) for run_statistics in classifier_statistics]
        plt.plot(x_values, y_values, linestyle="-", marker=".", label=classifier_name)
        plot_empty = False
    if not plot_empty:
        plt.legend()
    plt.savefig(report, format="pdf")
    plt.close()

def write_report(report_filename, num_threads, statistics):

    from matplotlib.backends.backend_pdf import PdfPages

    report = PdfPages(report_filename)

    write_statistic(report, statistics, f"Wall Clock Time (Train) :: {num_threads} thread(s)", "Time (s)", key="train-wall-clock-time")
    write_statistic(report, statistics, f"CPU Time (Train) :: {num_threads} thread(s)", "Time (s)", getter_func=lambda d, _: d["train-user-time"] + d["train-system-time"])
    write_statistic(report, statistics, f"Percent CPU (Train) :: {num_threads} thread(s)", "%", key="train-percent-cpu")
    write_statistic(report, statistics, f"Maximum RSS (Train) :: {num_threads} thread(s)", "RSS (GB)", getter_func=lambda d, _: d["train-max-rss"] / 1_000_000)
    write_statistic(report, statistics, f"Data Load Wall Clock Time", "Time (s)", key="train-data-load-time")
    write_statistic(report, statistics, f"Training Wall Clock Time", "Time (s)", key="train-training-time")
    write_statistic(report, statistics, f"Model Store Wall Clock Time", "Time (s)", key="train-model-store-time")
    write_statistic(report, statistics, f"Maximum Node Count", "No. of nodes", key="train-max-node-count")
    write_statistic(report, statistics, f"Maximum Tree Depth", "Levels", key="train-max-tree-depth")

    write_statistic(report, statistics, f"Wall Clock Time (Test) :: {num_threads} thread(s)", "Time (s)", key="test-wall-clock-time")
    write_statistic(report, statistics, f"CPU Time (Test) :: {num_threads} thread(s)", "Time (s)", getter_func=lambda d, _: d["test-user-time"] + d["test-system-time"])
    write_statistic(report, statistics, f"Percent CPU (Test) :: {num_threads} thread(s)", "%", key="test-percent-cpu")
    write_statistic(report, statistics, f"Maximum RSS (Test) :: {num_threads} thread(s)", "RSS (GB)", getter_func=lambda d, _: d["test-max-rss"] / 1_000_000)
    write_statistic(report, statistics, f"Model Load Wall Clock Time", "Time (s)", key="test-model-load-time")
    write_statistic(report, statistics, f"Data Load Wall Clock Time", "Time (s)", key="test-data-load-time")
    write_statistic(report, statistics, f"Classification Wall Clock Time", "Time (s)", key="test-classification-time")
    write_statistic(report, statistics, f"Label Store Wall Clock Time", "Time (s)", key="test-label-store-time")
    write_statistic(report, statistics, f"Accuracy", "Accuracy", key="test-accuracy")
    write_statistic(report, statistics, f"P4-metric", "P4-metric", key="test-P4-metric")
    write_statistic(report, statistics, f"Positive Predictive Value", "PPV", key="test-ppv")
    write_statistic(report, statistics, f"True Positive Rate", "True Positive Rate", key="test-tpr")
    write_statistic(report, statistics, f"True Negative Rate", "True Negative Rate", key="test-tnr")
    write_statistic(report, statistics, f"Negative Predictive Value", "NPV", key="test-npv")

    report.close()
