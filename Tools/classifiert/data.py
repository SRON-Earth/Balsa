import jsonpickle
import jsonpickle.ext.numpy
import numpy as np
import pathlib
import struct

jsonpickle.ext.numpy.register_handlers()

CACHE_DIR = pathlib.Path("cache")

def set_cache_dir(path):

    global CACHE_DIR
    CACHE_DIR = pathlib.Path(path)

def load_labelled_dataset_json(filename):

    with open(filename) as json_file:
        data_points, labels = jsonpickle.decode(json_file.read())
    assert data_points.dtype == np.float32
    assert labels.dtype == np.float64
    labels = labels.astype(np.float32)
    assert data_points.ndim == 2 and data_points.dtype == np.float32
    assert labels.ndim == 1 and labels.dtype == np.float32
    assert len(data_points) == len(labels)
    return data_points, labels

def load_dataset_bin(filename):

    header_format = "<I"
    header_size = struct.calcsize(header_format)
    with open(filename, "rb") as infile:
        num_columns, = struct.unpack(header_format, infile.read(header_size))
        dataset = np.frombuffer(infile.read(), "<f4")
        dataset.shape = (-1, num_columns)
    return dataset

def load_dataset_balsa(filename):

    header_format = "<4s4s4sI4sI4s"
    header_size = struct.calcsize(header_format)
    with open(filename, "rb") as infile:
        _, raw_value_type, _, num_rows, _, num_columns, _ = struct.unpack(header_format, infile.read(header_size))
        value_type = raw_value_type.decode("ascii", errors="ignore")
        if value_type == "fl32":
            dataset = np.frombuffer(infile.read(), "<f4")
        elif value_type == "ui08":
            dataset = np.frombuffer(infile.read(), "<u1")
        else:
            raise RuntimeError("Unsupported value type: '" + value_type + "'.")
    dataset.shape = (-1, num_columns)
    assert len(dataset) == num_rows, "The number of rows read does not match the row count stored in the header."
    return dataset

def store_labelled_dataset_csv(filename, data_points, labels):

    assert data_points.ndim == 2 and data_points.dtype == np.float32
    assert labels.ndim == 1 and labels.dtype == np.float32
    assert len(data_points) == len(labels)

    num_rows, num_features = len(data_points), data_points.shape[-1]
    with open(filename, "w") as outfile:
        header = [f"feature-{i}" for i in range(num_features)] + ["label"]
        outfile.write(",".join(header) + "\n")
        for i in range(num_rows):
            outfile.write(",".join([f"{value:.16f}" for value in data_points[i]] + [str(int(labels[i]))]) + "\n")

def store_dataset_bin(filename, dataset):

    assert dataset.ndim == 1 or dataset.ndim == 2
    assert dataset.dtype == np.float32

    num_rows, num_columns = (*dataset.shape, 1)[:2]
    with open(filename, "wb") as outfile:
        outfile.write(struct.pack("<I", num_columns))
        for i in range(num_rows):
            outfile.write(dataset[i].tobytes())

def store_labelled_dataset_bin(data_filename, label_filename, data_points, labels):

    assert data_points.ndim == 2 and data_points.dtype == np.float32
    assert labels.ndim == 1 and labels.dtype == np.float32
    assert len(data_points) == len(labels)

    store_dataset_bin(data_filename, data_points)
    store_dataset_bin(label_filename, labels)

def store_dataset_balsa(filename, dataset):

    assert dataset.ndim == 1 or dataset.ndim == 2
    assert dataset.dtype == np.float32

    num_rows, num_columns = (*dataset.shape, 1)[:2]
    with open(filename, "wb") as outfile:
        outfile.write(b"tabl")
        outfile.write(b"fl32")
        outfile.write(b"rows")
        outfile.write(struct.pack("<I", num_rows))
        outfile.write(b"cols")
        outfile.write(struct.pack("<I", num_columns))
        outfile.write(b"data")
        for i in range(num_rows):
            outfile.write(dataset[i].tobytes())

def store_labelled_dataset_balsa(data_filename, label_filename, data_points, labels):

    assert data_points.ndim == 2 and data_points.dtype == np.float32
    assert labels.ndim == 1 and labels.dtype == np.float32
    assert len(data_points) == len(labels)

    store_dataset_balsa(data_filename, data_points)
    store_dataset_balsa(label_filename, labels)

def store_labelled_dataset(data_format, data_filename, label_filename, data_points, labels):

    if label_filename and data_format == "csv":
        raise RuntimeError("The \"csv\" format stores both data and labels in a single file.")

    if not label_filename and data_format != "csv":
        raise RuntimeError("No filename specified for the label file.")

    if data_format == "csv":
        store_labelled_dataset_csv(data_filename, data_points, labels)
    elif data_format == "bin":
        store_labelled_dataset_bin(data_filename, label_filename, data_points, labels)
    elif data_format == "balsa":
        store_labelled_dataset_balsa(data_filename, label_filename, data_points, labels)
    else:
        raise RuntimeError("Unsupported data format: " + str(data_format) + ".")

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

    for data_format in ("csv", "bin", "balsa"):
        filenames = get_train_dataset_filenames(data_size, data_format)
        if not all(filename is None or filename.is_file() for filename in filenames):
            return False
    return True

def remove_from_cache(data_size):

    for data_format in ("csv", "bin", "balsa"):
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
            data_points, labels = load_labelled_dataset_json(train_data_filename)
            assert np.sum(labels == 0.0) + np.sum(labels == 1.0) == labels.size

        new_data_points, new_labels = sample_dataset(data_points, labels, data_size, random_generator=random_generator)

        for data_format in ("csv", "bin", "balsa"):
            train_data_filename, train_label_filename = get_train_dataset_filenames(data_size, data_format)
            store_labelled_dataset(data_format, train_data_filename, train_label_filename, new_data_points, new_labels)

def ingest_test_dataset(test_data_filename):

    data_points, labels = load_labelled_dataset_json(test_data_filename)

    for data_format in ("csv", "bin", "balsa"):
        test_data_filename, test_label_filename = get_test_dataset_filenames(data_format)
        store_labelled_dataset(data_format, test_data_filename, test_label_filename, data_points, labels)
