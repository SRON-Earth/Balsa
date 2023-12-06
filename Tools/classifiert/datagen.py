import jsonpickle
import jsonpickle.ext.numpy
import numpy as np
import pathlib
import struct

jsonpickle.ext.numpy.register_handlers()

CACHE_DIR = pathlib.Path("cache")


def load_dataset_json(filename):

    with open(filename) as json_file:
        data_points, labels = jsonpickle.decode(json_file.read())
    assert data_points.dtype == np.float32
    assert labels.dtype == np.float64
    labels = labels.astype(np.float32)
    assert data_points.ndim == 2 and data_points.dtype == np.float32
    assert labels.ndim == 1 and labels.dtype == np.float32
    assert len(data_points) == len(labels)
    return data_points, labels


def store_dataset_csv(filename, data_points, labels):

    assert data_points.ndim == 2 and data_points.dtype == np.float32
    assert labels.ndim == 1 and labels.dtype == np.float32
    assert len(data_points) == len(labels)

    num_features = data_points.shape[-1]
    with open(filename, "w") as outf:
        header = [f"feature-{i}" for i in range(num_features)] + ["label"]
        outf.write(",".join(header) + "\n")
        for i in range(len(data_points)):
            outf.write(",".join([f"{value:.16f}" for value in data_points[i]] + [str(int(labels[i]))]) + "\n")


def store_dataset_bin(data_filename, label_filename, data_points, labels):

    assert data_points.ndim == 2 and data_points.dtype == np.float32
    assert labels.ndim == 1 and labels.dtype == np.float32
    assert len(data_points) == len(labels)

    num_columns = data_points.shape[-1]
    with open(data_filename, "wb") as outf:
        outf.write(struct.pack("<I", num_columns))
        for i in range(len(data_points)):
            outf.write(data_points[i].tobytes())

    num_columns = 1
    with open(label_filename, "wb") as outf:
        outf.write(struct.pack("<I", num_columns))
        for i in range(len(labels)):
            outf.write(labels[i].tobytes())


def get_train_dataset_filenames(data_size, data_format):

    train_data_file = pathlib.Path(CACHE_DIR / f"train-data-{data_size}.{data_format}").absolute()
    if data_format == "csv":
        return train_data_file, None
    train_label_file = pathlib.Path(CACHE_DIR / f"train-label-{data_size}.{data_format}").absolute()
    return train_data_file, train_label_file


def get_test_dataset_filenames(data_format):

    train_data_file = pathlib.Path(CACHE_DIR / f"test-data.{data_format}").absolute()
    if data_format == "csv":
        return train_data_file, None
    train_label_file = pathlib.Path(CACHE_DIR / f"test-label.{data_format}").absolute()
    return train_data_file, train_label_file


def is_cached(data_size):

    for data_format in ("csv", "bin"):
        filenames = get_train_dataset_filenames(data_size, data_format)
        if not all(filename is None or filename.is_file() for filename in filenames):
            return False
    return True


def remove_from_cache(data_size):

    for data_format in ("csv", "bin"):
        for filename in get_train_dataset_filenames(data_size, data_format):
            if filename is not None:
                filename.unlink(missing_ok=True)


def sample_dataset(data_points, labels, data_size, *, random_generator=None, replace=False):

    assert len(data_points) == len(labels)

    if random_generator is None:
        random_generator = np.random.default_rng()

    assert (data_size <= len(data_points)) or replace
    index = random_generator.choice(len(data_points), data_size, replace=replace)
    return data_points[index], labels[index]


def generate_train_datasets(train_data_filename, data_sizes, *, use_cache=True, seed=None):

    random_generator = np.random.default_rng(seed)

    data_points, labels = None, None
    for data_size in data_sizes:

        in_cache = is_cached(data_size)

        if in_cache and use_cache:
            print("\033[32m" + f"{data_size} [cached]" + "\033[0m")
            continue

        if in_cache:
            print("\033[32m" + f"{data_size} [forced]" + "\033[0m")
        else:
            print("\033[32m" + f"{data_size}" + "\033[0m")

        remove_from_cache(data_size)

        if data_points is None:
            assert labels is None
            data_points, labels = load_dataset_json(train_data_filename)
            assert np.sum(labels == 0.0) + np.sum(labels == 1.0) == labels.size

        new_data_points, new_labels = sample_dataset(data_points, labels, data_size, random_generator=random_generator)

        train_data_filename, train_label_filename = get_train_dataset_filenames(data_size, "csv")
        store_dataset_csv(train_data_filename, new_data_points, new_labels)

        train_data_filename, train_label_filename = get_train_dataset_filenames(data_size, "bin")
        store_dataset_bin(train_data_filename, train_label_filename, new_data_points, new_labels)


def ingest_test_dataset(test_data_filename):

    data_points, labels = load_dataset_json(test_data_filename)

    test_data_filename, test_label_filename = get_test_dataset_filenames("csv")
    store_dataset_csv(test_data_filename, data_points, labels)

    test_data_filename, test_label_filename = get_test_dataset_filenames("bin")
    store_dataset_bin(test_data_filename, test_label_filename, data_points, labels)
