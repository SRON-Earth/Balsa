import pathlib

from .. import register_classifier
from ..util import run_program, get_statistics_from_time_file

PACKAGE_DATA_PATH = pathlib.Path(__file__).parent.parent.absolute() / "data"


def run(run_path, train_data_filename, train_label_filename, test_data_filename, test_label_filename, *,
        num_estimators, max_tree_depth, num_threads):

    TIME_FILE = "time.txt"

    args = ["-e", str(num_estimators), "-t", str(num_threads)]
    if max_tree_depth is not None:
        args += ["-d", str(max_tree_depth)]
    args += [str(train_data_filename), str(train_label_filename)]
    args += [str(test_data_filename), str(test_label_filename)]
    args += ["lightgbm.model"]

    result = run_program(PACKAGE_DATA_PATH / "lightgbm-train.py", *args, log=True, time_file=TIME_FILE, cwd=run_path)

    run_statistics = {}
    for line in result.stdout.split("\n"):
        if "max-tree-depth" in line:
            run_statistics["depth"] = int(line.split()[-1])
        if "max-node-count" in line:
            run_statistics["node_count"] = int(line.split()[-1])
        if "test-accuracy" in line:
            run_statistics["accuracy"] = float(line.split()[-1])

    run_statistics.update(get_statistics_from_time_file(run_path / TIME_FILE))

    return run_statistics

register_classifier("lightgbm", "bin", run)
