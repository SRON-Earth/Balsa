import numpy as np

from .. import register_classifier
from ..util import run_program, get_statistics_from_time_file

def run(run_path, train_file, test_file, threads):

    run_statistics = {}
    time_file = "time.txt"
    result = run_program("rftrain",
            str(train_file),
            "jigsaw.forest",
            time_file=time_file,
            cwd=run_path)
    run_statistics.update(get_statistics_from_time_file(run_path / time_file))

    return run_statistics

register_classifier("jigsaw", "bin", run)
