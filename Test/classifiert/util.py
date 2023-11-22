import pathlib
import subprocess


def run_program(program, *args, log=False, time_file=None, timeout=None, cwd=None):

    program = pathlib.Path(program)
    if time_file is not None:
        result = subprocess.run(["time", "-v", "-o", time_file, str(program), *args], capture_output=True, text=True,
                                timeout=timeout, cwd=cwd)
    else:
        result = subprocess.run([str(program), *args], capture_output=True, text=True, timeout=timeout, cwd=cwd)
    if log:
        stdout_filename = f"{program.name}-stdout.txt"
        stdout_path = stdout_filename if cwd is None else cwd / stdout_filename
        with open(stdout_path, "w") as outputf:
            outputf.write(result.stdout)
        stderr_filename = f"{program.name}-stderr.txt"
        stderr_path = stderr_filename if cwd is None else cwd / stderr_filename
        with open(stderr_path, "w") as outputf:
            outputf.write(result.stderr)
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
