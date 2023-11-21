import jsonpickle
import jsonpickle.ext.numpy
import numpy as np
import pathlib
import struct

jsonpickle.ext.numpy.register_handlers()

RND_GENERATOR = np.random.default_rng()
CACHE_DIR     = pathlib.Path("cache")
DATASET       = pathlib.Path("xy_300_post_0.001.json")


def store_dataset_csv(filename, data_points_false, data_points_true):

    assert data_points_false.shape == data_points_true.shape

    num_features = data_points_false.shape[-1]
    with open(filename, "w") as outf:
        header = [f"feature-{i}" for i in range(num_features)] + ["label"]
        outf.write(",".join(header) + "\n")
        for i in range(len(data_points_false)):
            outf.write(",".join([f"{value:.16f}" for value in data_points_false[i]] + ["0"]) + "\n")
        for i in range(len(data_points_true)):
            outf.write(",".join([f"{value:.16f}" for value in data_points_true[i]] + ["1"]) + "\n")


def store_dataset_bin(data_filename, label_filename, data_points_false, data_points_true):

    assert data_points_false.shape == data_points_true.shape

    num_columns = data_points_false.shape[-1]
    with open(data_filename, "wb") as outf:
        outf.write(struct.pack("<I", num_columns))
        for i in range(len(data_points_false)):
            outf.write(data_points_false[i].tobytes())
        for i in range(len(data_points_true)):
            outf.write(data_points_true[i].tobytes())

    num_columns = 1
    with open(label_filename, "wb") as outf:
        outf.write(struct.pack("<I", num_columns))
        for i in range(len(data_points_false)):
            outf.write(struct.pack("<f", 0.0))
        for i in range(len(data_points_true)):
            outf.write(struct.pack("<f", 1.0))


def get_dataset_filenames(data_size, test_percentage, data_format):

    train_data_file = pathlib.Path(CACHE_DIR / f"train-data-{data_size}-{test_percentage}.{data_format}").absolute()
    test_data_file  = pathlib.Path(CACHE_DIR / f"test-data-{data_size}-{test_percentage}.{data_format}" ).absolute()

    if data_format == "csv":
        return train_data_file, None, test_data_file, None

    train_label_file = pathlib.Path(CACHE_DIR / f"train-label-{data_size}-{test_percentage}.{data_format}").absolute()
    test_label_file  = pathlib.Path(CACHE_DIR / f"test-label-{data_size}-{test_percentage}.{data_format}" ).absolute()
    return train_data_file, train_label_file, test_data_file, test_label_file


def is_cached(data_size, test_percentage):

    for data_format in ("csv", "bin"):
        filenames = get_dataset_filenames(data_size, test_percentage, data_format)
        if not all(filename is None or filename.is_file() for filename in filenames):
            return False
    return True


def generate_datasets(data_sizes, test_percentage, use_cache=True):

    data_points, labels = None, None

    for data_size in data_sizes:

        in_cache = is_cached(data_size, test_percentage)

        if in_cache and use_cache:
            print("\033[32m" + f"{data_size} [cached]" + "\033[0m")
            continue

        if in_cache:
            print("\033[32m" + f"{data_size} [forced]" + "\033[0m")
        else:
            print("\033[32m" + f"{data_size}" + "\033[0m")

        for data_format in ("csv", "bin"):
            for filename in get_dataset_filenames(data_size, test_percentage, data_format):
                if filename is not None:
                    filename.unlink(missing_ok=True)

        if data_points is None or labels is None:
            with open(DATASET) as json_file:
                data_points, labels = jsonpickle.decode(json_file.read())
            assert np.sum(labels == 0.0) + np.sum(labels == 1.0) == labels.size
            assert data_points.dtype == np.float32
            print(labels.dtype)
            labels = labels.astype(np.float32)
            assert labels.dtype == np.float32

        num_data_points, num_features = data_points.shape
        assert data_size <= num_data_points
        assert num_data_points % 2 == 0
        assert data_size % 2 == 0

        index_false = np.flatnonzero(labels == 0)
        index_true  = np.flatnonzero(labels == 1)
        assert len(index_false) == len(index_true)

        data_points_false = data_points[RND_GENERATOR.choice(index_false, data_size // 2, replace=False)]
        data_points_true  = data_points[RND_GENERATOR.choice(index_true , data_size // 2, replace=False)]
        assert len(data_points_false) == data_size // 2
        assert len(data_points_true) == data_size // 2

        test_size = 2 * round(data_size // 2 * test_percentage / 100)
        training_size = data_size - test_size

        for data_format in ("csv", "bin"):
            train_data_filename, train_label_filename, test_data_filename, test_label_filename = \
                get_dataset_filenames(data_size, test_percentage, data_format)
            if data_format == "csv":
                store_dataset_csv(train_data_filename, data_points_false[:training_size//2], data_points_true[:training_size//2])
                store_dataset_csv(test_data_filename, data_points_false[training_size//2:], data_points_true[training_size//2:])
            elif data_format == "bin":
                store_dataset_bin(train_data_filename, train_label_filename, data_points_false[:training_size//2], data_points_true[:training_size//2])
                store_dataset_bin(test_data_filename, test_label_filename, data_points_false[training_size//2:], data_points_true[training_size//2:])
            else:
                raise RuntimeError
