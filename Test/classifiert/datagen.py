import jsonpickle
import jsonpickle.ext.numpy
import numpy as np
import pathlib
import struct

jsonpickle.ext.numpy.register_handlers()

RND_GENERATOR = np.random.default_rng()
CACHE_DIR     = pathlib.Path("cache")


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


def generate_train_datasets(train_data_filename, data_sizes, use_cache=True):

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

        if data_points is None or labels is None:
            data_points, labels = load_dataset_json(train_data_filename)
            with open(train_data_filename) as json_file:
                data_points, labels = jsonpickle.decode(json_file.read())
            assert np.sum(labels == 0.0) + np.sum(labels == 1.0) == labels.size

        num_data_points, num_features = data_points.shape
        assert data_size <= num_data_points
        assert num_data_points % 2 == 0
        assert data_size % 2 == 0

        index_false = np.flatnonzero(labels == 0)
        index_true  = np.flatnonzero(labels == 1)
        assert len(index_false) == len(index_true)

        selected_data_points_false = data_points[RND_GENERATOR.choice(index_false, data_size // 2, replace=False)]
        selected_data_points_true  = data_points[RND_GENERATOR.choice(index_true , data_size // 2, replace=False)]
        assert len(selected_data_points_false) == data_size // 2
        assert len(selected_data_points_true) == data_size // 2

        selected_data_points = np.vstack((selected_data_points_false, selected_data_points_true))
        selected_labels = np.concatenate((np.zeros(data_size // 2, dtype=np.float32), np.ones(data_size // 2, dtype=np.float32)))

        train_data_filename, train_label_filename = get_train_dataset_filenames(data_size, "csv")
        store_dataset_csv(train_data_filename, selected_data_points, selected_labels)

        train_data_filename, train_label_filename = get_train_dataset_filenames(data_size, "bin")
        store_dataset_bin(train_data_filename, train_label_filename, selected_data_points, selected_labels)


def ingest_test_dataset(test_data_filename):

    data_points, labels = load_dataset_json(test_data_filename)

    test_data_filename, test_label_filename = get_test_dataset_filenames("csv")
    store_dataset_csv(test_data_filename, data_points, labels)

    test_data_filename, test_label_filename = get_test_dataset_filenames("bin")
    store_dataset_bin(test_data_filename, test_label_filename, data_points, labels)
