import matplotlib.pyplot as plt


def write_report(report_filename, data_sizes, test_percentage, num_threads, statistics_):

    plt.figure()
    plt.title(f"CPU Time :: {num_threads} thread(s)")
    plt.xlabel("No. of samples")
    plt.ylabel("CPU Time (s)")
    for classifier, statistics in statistics_.items():
        num_samples = [stat["data_size"] - 2 * round(stat["data_size"] / 2 * test_percentage / 100) for stat in statistics]
        cpu_time = [stat["user_time"] + stat["system_time"] for stat in statistics]
        plt.plot(num_samples, cpu_time, linestyle="-", marker=".", label=classifier)
    plt.legend()

    plt.figure()
    plt.title(f"Maximum RSS :: {num_threads} thread(s)")
    plt.xlabel("No. of samples")
    plt.ylabel("Maximum RSS (MB)")
    for classifier, statistics in statistics_.items():
        num_samples = [stat["data_size"] - 2 * round(stat["data_size"] / 2 * test_percentage / 100) for stat in statistics]
        max_rss = [stat["max_rss"] / 1000 for stat in statistics]
        plt.plot(num_samples, max_rss, linestyle="-", marker=".", label=classifier)
    plt.legend()

    plt.figure()
    plt.title(f"Accuracy :: {num_threads} thread(s)")
    plt.xlabel("No. of samples")
    plt.ylabel("Accuracy (%)")
    for classifier, statistics in statistics_.items():
        num_samples = [stat["data_size"] - 2 * round(stat["data_size"] / 2 * test_percentage / 100) for stat in statistics]
        accuracy = [round(stat.get("accuracy", 0) * 100) for stat in statistics]
        plt.plot(num_samples, accuracy, linestyle="-", marker=".", label=classifier)
    plt.legend()

    plt.figure()
    plt.title(f"Maximum node count :: {num_threads} thread(s)")
    plt.xlabel("No. of samples")
    plt.ylabel("No. of nodes")
    for classifier, statistics in statistics_.items():
        num_samples = [stat["data_size"] - 2 * round(stat["data_size"] / 2 * test_percentage / 100) for stat in statistics]
        node_count = [stat.get("node_count", 0) for stat in statistics]
        plt.plot(num_samples, node_count, linestyle="-", marker=".", label=classifier)
    plt.legend()

    plt.figure()
    plt.title(f"Maximum depth :: {num_threads} thread(s)")
    plt.xlabel("No. of samples")
    plt.ylabel("Depth")
    for classifier, statistics in statistics_.items():
        num_samples = [stat["data_size"] - 2 * round(stat["data_size"] / 2 * test_percentage / 100) for stat in statistics]
        depth = [stat.get("depth", 0) for stat in statistics]
        plt.plot(num_samples, depth, linestyle="-", marker=".", label=classifier)
    plt.legend()

    from matplotlib.backends.backend_pdf import PdfPages
    report = PdfPages(report_filename)
    for fignum in plt.get_fignums():
        figure = plt.figure(fignum)
        figure.savefig(report, format="pdf")
        plt.close(fignum)
    report.close()
