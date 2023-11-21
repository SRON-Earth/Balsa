import numpy as np

from .. import register_classifier
from ..util import run_program, get_statistics_from_time_file

def run(run_path, train_data_filename, train_label_filename, test_data_filename, test_label_filename,
        num_estimators, max_tree_depth, num_threads):

    run_statistics = {}
    time_file = "time.txt"

    args = ["-c", str(num_estimators), "-t", str(num_threads)]
    if max_tree_depth is not None:
        args += ["-d", str(max_tree_depth)]

    result = run_program("rftrain",
            *args,
            str(train_data_filename),
            str(train_label_filename),
            "jigsaw.forest",
            time_file=time_file,
            cwd=run_path)
    run_statistics.update(get_statistics_from_time_file(run_path / time_file))

    return run_statistics

register_classifier("jigsaw", "bin", run)
