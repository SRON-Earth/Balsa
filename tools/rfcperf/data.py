import jsonpickle
import jsonpickle.ext.numpy
import numpy as np
import pathlib
import struct

jsonpickle.ext.numpy.register_handlers()

CACHE_DIR = pathlib.Path("cache")
DATA_FORMATS = ("csv", "bin", "balsa")

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

def read_string(infile):

    fmt = "<B"
    fmt_size = struct.calcsize(fmt)
    string_size, = struct.unpack(fmt, infile.read(fmt_size))
    return infile.read(string_size).decode("ascii")

def read_kv_pair(infile):

    key = read_string(infile)

    raw_value_type = infile.read(4)

    if raw_value_type == b"strn":
        value = read_string(infile)

    else:
        if raw_value_type == b"ui08":
            fmt = "<B"
        elif raw_value_type == b"ui32":
            fmt = "<I"
        elif raw_value_type == b"fl32":
            fmt = "<f"
        elif raw_value_type == b"fl64":
            fmt = "<d"
        else:
            assert False

        fmt_size = struct.calcsize(fmt)
        value, = struct.unpack(fmt, infile.read(fmt_size))

    return key, value

def read_dictionary(infile):

    dict_start_marker = infile.read(4)
    assert dict_start_marker == b"dict"

    fmt = "<B"
    fmt_size = struct.calcsize(fmt)
    dict_size, = struct.unpack(fmt, infile.read(fmt_size))

    result = {}
    for i in range(dict_size):
        key, value = read_kv_pair(infile)
        result[key] = value

    dict_end_marker = infile.read(4)
    assert dict_end_marker == b"tcid"

    return result

def load_dataset_balsa(filename):

    with open(filename, "rb") as infile:

        file_signature = infile.read(4)
        assert file_signature == b"blsa"
        endianness_marker = infile.read(4)
        assert endianness_marker == b"lend"

        file_header = read_dictionary(infile)
        assert file_header["file_major_version"] == 1
        assert file_header["file_minor_version"] == 0

        table_start_marker = infile.read(4)
        assert table_start_marker == b"tabl"

        table_header = read_dictionary(infile)
        num_rows = table_header["row_count"]
        num_columns = table_header["column_count"]
        scalar_type_id = table_header["scalar_type_id"]

        if scalar_type_id == "fl32":
            size = num_rows * num_columns * 4
            dataset = np.frombuffer(infile.read(size), "<f4")
        elif scalar_type_id == "ui08":
            size = num_rows * num_columns * 1
            dataset = np.frombuffer(infile.read(size), "<u1")
        else:
            raise RuntimeError("Unsupported value type: '" + scalar_type_id + "'.")

        dataset.shape = (-1, num_columns)
        assert len(dataset) == num_rows, "The number of rows read does not match the row count stored in the header."

        table_end_marker = infile.read(4)
        assert table_end_marker == b"lbat"

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

def write_string(outfile, string):

    raw_string = string.encode("ascii")
    assert len(raw_string) < 256
    outfile.write(struct.pack(f"<B{len(raw_string)}s", len(raw_string), raw_string))

def write_kv_pair(outfile, key, value, value_type):

    if value_type == "strn":
        write_string(outfile, key)
        outfile.write(b"strn")
        write_string(outfile, value)
        return

    if value_type == "ui08":
        fmt = "<B"
    elif value_type == "ui32":
        fmt = "<I"
    elif value_type == "fl32":
        fmt = "<f"
    elif value_type == "fl64":
        fmt = "<d"
    else:
        assert False

    write_string(outfile, key)
    outfile.write(value_type.encode("ascii"))
    outfile.write(struct.pack(fmt, value))

def store_dataset_balsa(filename, dataset):

    assert dataset.ndim == 1 or dataset.ndim == 2
    assert dataset.dtype == np.float32

    num_rows, num_columns = (*dataset.shape, 1)[:2]
    with open(filename, "wb") as outfile:

        # Write file signature.
        outfile.write(b"blsa")

        # Write endianness marker.
        outfile.write(b"lend")

        # Write file header dictionary.
        outfile.write(b"dict")
        outfile.write(struct.pack("<B", 2))
        write_kv_pair(outfile, "file_major_version", 1, "ui08")
        write_kv_pair(outfile, "file_minor_version", 0, "ui08")
        outfile.write(b"tcid")

        # Write table start marker.
        outfile.write(b"tabl")

        # Write table header dictionary.
        outfile.write(b"dict")
        outfile.write(struct.pack("<B", 3))
        write_kv_pair(outfile, "row_count", num_rows, "ui32")
        write_kv_pair(outfile, "column_count", num_columns, "ui32")
        write_kv_pair(outfile, "scalar_type_id", "fl32", "strn")
        outfile.write(b"tcid")

        # Write table elements.
        for i in range(num_rows):
            outfile.write(dataset[i].tobytes())

        # Write table end marker.
        outfile.write(b"lbat")

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

def get_dataset_filenames(purpose, data_format, data_size, test_percentage):

    suffix = "." + data_format
    if test_percentage is not None:
        suffix = "-oob-" + str(test_percentage) + suffix
    if data_size is not None:
        suffix = "-" + str(data_size) + suffix
    data_file = pathlib.Path(CACHE_DIR / (purpose + "-data" + suffix)).absolute()
    label_file = pathlib.Path(CACHE_DIR / (purpose + "-label" + suffix)).absolute()
    return (data_file, None) if data_format == "csv" else (data_file, label_file)

def get_train_dataset_filenames(data_format, data_size, test_percentage=None):

    return get_dataset_filenames("train", data_format, data_size, test_percentage)

def get_test_dataset_filenames(data_format, data_size=None, test_percentage=None):

    return get_dataset_filenames("test", data_format, data_size, test_percentage)

def is_cached(data_size, test_percentage):

    for data_format in DATA_FORMATS:
        filenames = get_train_dataset_filenames(data_format, data_size, test_percentage)
        if test_percentage is not None:
            filenames += get_test_dataset_filenames(data_format, data_size, test_percentage)
        for filename in filenames:
            if filename is not None and not filename.is_file():
                return False
    return True

def remove_from_cache(data_size, test_percentage):

    for data_format in DATA_FORMATS:
        filenames = get_train_dataset_filenames(data_format, data_size, test_percentage)
        if test_percentage is not None:
            filenames += get_test_dataset_filenames(data_format, data_size, test_percentage)
        for filename in filenames:
            if filename is not None:
                filename.unlink(missing_ok=True)

def sample_dataset(data_points, labels, data_size, *, random_generator=None, replace=False):

    assert len(data_points) == len(labels)

    if random_generator is None:
        random_generator = np.random.default_rng()

    assert (data_size <= len(data_points)) or replace
    index = random_generator.choice(len(data_points), data_size, replace=replace)
    return data_points[index], labels[index]

def generate_datasets(train_data_filename, data_sizes, *, use_cache=True, test_percentage=None, seed=None):

    random_generator = np.random.default_rng(seed)

    data_points, labels = None, None
    for data_size in data_sizes:

        in_cache = is_cached(data_size, test_percentage)

        if in_cache and use_cache and seed is None:
            print("\033[32m" + f"{data_size} [cached]" + "\033[0m")
            continue

        if in_cache:
            print("\033[32m" + f"{data_size} [forced]" + "\033[0m")
        else:
            print("\033[32m" + f"{data_size}" + "\033[0m")

        remove_from_cache(data_size, test_percentage)

        if data_points is None:
            assert labels is None
            data_points, labels = load_labelled_dataset_json(train_data_filename)
            assert np.sum(labels == 0.0) + np.sum(labels == 1.0) == labels.size

        test_size = 0 if test_percentage is None else round(test_percentage * data_size / 100.0)
        new_data_points, new_labels = sample_dataset(data_points, labels, data_size + test_size, random_generator=random_generator)

        for data_format in DATA_FORMATS:
            train_data_filename, train_label_filename = get_train_dataset_filenames(data_format, data_size, test_percentage)
            store_labelled_dataset(data_format, train_data_filename, train_label_filename, new_data_points[:data_size], new_labels[:data_size])

        if test_percentage is None:
            continue

        for data_format in DATA_FORMATS:
            test_data_filename, test_label_filename = get_test_dataset_filenames(data_format, data_size, test_percentage)
            store_labelled_dataset(data_format, test_data_filename, test_label_filename, new_data_points[data_size:], new_labels[data_size:])

def ingest_test_dataset(test_data_filename):

    data_points, labels = load_labelled_dataset_json(test_data_filename)

    for data_format in DATA_FORMATS:
        test_data_filename, test_label_filename = get_test_dataset_filenames(data_format)
        store_labelled_dataset(data_format, test_data_filename, test_label_filename, data_points, labels)
