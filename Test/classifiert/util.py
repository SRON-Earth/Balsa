import pathlib
import subprocess


def run_program(program, *args, time_file=None, timeout=None, cwd=None):

    if time_file is not None:
        result = subprocess.run(["time", "-v", "-o", time_file, program, *args], capture_output=True, text=True, timeout=timeout, cwd=cwd)
    else:
        result = subprocess.run([program, *args], capture_output=True, text=True, timeout=timeout, cwd=cwd)
    assert result.returncode == 0, f"Program '{program}' failed with exit code: {result.returncode}"
    return result


def get_statistics_from_time_file(time_file):

    statistics = {}
    for line in pathlib.Path(time_file).read_text().split("\n"):
        if "User time" in line:
            statistics["user_time"] = float(line.split()[-1])
        elif "System time" in line:
            statistics["system_time"] = float(line.split()[-1])
        elif "Maximum resident set size" in line:
            statistics["max_rss"] = int(line.split()[-1])
    return statistics
