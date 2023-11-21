import jsonpickle
import jsonpickle.ext.numpy
import numpy as np
import pathlib
import struct

jsonpickle.ext.numpy.register_handlers()

RND_GENERATOR = np.random.default_rng()
CACHE_DIR     = pathlib.Path("cache")
DATASET       = pathlib.Path("xy_300_post_0.001.json")


def store_dataset_csv(filename, points_false, points_true):

    assert points_false.shape == points_true.shape
    num_features = points_false.shape[-1]
    with open(filename, "w") as outf:
        header = [f"feature-{i}" for i in range(num_features)] + ["label"]
        outf.write(",".join(header) + "\n")
        for i in range(len(points_false)):
            outf.write(",".join([f"{value:.16f}" for value in points_false[i]] + ["0"]) + "\n")
        for i in range(len(points_true)):
            outf.write(",".join([f"{value:.16f}" for value in points_true[i]] + ["1"]) + "\n")


def store_dataset_bin(filename, points_false, points_true):

    assert points_false.shape == points_true.shape
    num_columns = points_false.shape[-1] + 1
    with open(filename, "wb") as outf:
        outf.write(struct.pack("<I", num_columns))
        for i in range(len(points_false)):
            outf.write(points_false[i].tobytes())
            outf.write(struct.pack("<f", 0.0))
        for i in range(len(points_true)):
            outf.write(points_true[i].tobytes())
            outf.write(struct.pack("<f", 1.0))


def get_dataset_filenames(data_size, test_percentage, data_format):

    train_file = pathlib.Path(CACHE_DIR / f"train-{data_size}-{test_percentage}.{data_format}")
    test_file = pathlib.Path(CACHE_DIR / f"test-{data_size}-{test_percentage}.{data_format}")
    return train_file, test_file


def generate_datasets(data_sizes, test_percentage, use_cache=True):

    points, labels = None, None

    for data_size in data_sizes:

        in_cache = True
        for data_format in ("csv", "bin"):
            train_file, test_file = get_dataset_filenames(data_size, test_percentage, data_format)
            if not train_file.is_file() or not test_file.is_file():
                in_cache = False
                break

        if in_cache and use_cache:
            print("\033[32m" + f"{data_size} [cached]" + "\033[0m")
            continue

        if in_cache:
            print("\033[32m" + f"{data_size} [forced]" + "\033[0m")
        else:
            print("\033[32m" + f"{data_size}" + "\033[0m")

        for data_format in ("csv", "bin"):
            train_file, test_file = get_dataset_filenames(data_size, test_percentage, data_format)
            train_file.unlink(missing_ok=True)
            test_file.unlink(missing_ok=True)

        if points is None or labels is None:

            with open(DATASET) as json_file:
                points, labels = jsonpickle.decode(json_file.read())
            assert (np.sum(labels == 0) + np.sum(labels == 1)) == labels.size
            labels = labels.astype(int)

        num_points, num_features = points.shape
        assert data_size <= num_points
        assert num_points % 2 == 0
        assert data_size % 2 == 0

        index_false = np.flatnonzero(labels == 0)
        index_true  = np.flatnonzero(labels == 1)
        assert len(index_false) == len(index_true)

        dataset_false = points[RND_GENERATOR.choice(index_false, data_size//2, replace=False)]
        dataset_true  = points[RND_GENERATOR.choice(index_true , data_size//2, replace=False)]
        assert len(dataset_false) == data_size // 2
        assert len(dataset_true) == data_size // 2

        test_size = 2 * round(data_size / 2 * test_percentage / 100)
        training_size = data_size - test_size

        for data_format in ("csv", "bin"):
            train_file, test_file = get_dataset_filenames(data_size, test_percentage, data_format)
            if data_format == "csv":
                store_dataset_csv(train_file, dataset_false[:training_size//2], dataset_true[:training_size//2])
                store_dataset_csv(test_file , dataset_false[training_size//2:], dataset_true[training_size//2:])
            elif data_format == "bin":
                store_dataset_bin(train_file, dataset_false[:training_size//2], dataset_true[:training_size//2])
                store_dataset_bin(test_file , dataset_false[training_size//2:], dataset_true[training_size//2:])
            else:
                raise RuntimeError
