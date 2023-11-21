import pathlib

from .. import register_classifier
from ..util import run_program, get_statistics_from_time_file

PACKAGE_DATA_PATH = pathlib.Path(__file__).parent.parent.absolute() / "data"

def run(run_path, train_data_filename, train_label_filename, test_data_filename, test_label_filename,
        num_estimators, max_tree_depth, num_threads):

    run_statistics = {}
    time_file = "time.txt"
    result = run_program(PACKAGE_DATA_PATH / "lightgbm-train.py",
                         str(train_data_filename),
                         str(train_label_filename),
                         str(test_data_filename),
                         str(test_label_filename),
                         str(num_estimators),
                         str(max_tree_depth),
                         str(num_threads),
                         time_file=time_file,
                         cwd=run_path)

    for line in result.stdout.split("\n"):
        if "max-tree-depth" in line:
            run_statistics["depth"] = int(line.split()[-1])
        if "max-node-count" in line:
            run_statistics["node_count"] = int(line.split()[-1])
        if "test-accuracy" in line:
            run_statistics["accuracy"] = float(line.split()[-1])

    run_statistics.update(get_statistics_from_time_file(run_path / time_file))

    return run_statistics

register_classifier("lightgbm", "bin", run)
