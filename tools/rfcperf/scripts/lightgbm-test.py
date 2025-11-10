import argparse
import lightgbm as lgb
import numpy as np
import pathlib
import struct
import time

def load_dataset_bin(filename):

    header_format = "<I"
    header_size = struct.calcsize(header_format)
    with open(filename, "rb") as infile:
        num_columns, = struct.unpack(header_format, infile.read(header_size))
        dataset = np.frombuffer(infile.read(), "<f4")
        dataset.shape = (-1, num_columns)
    return dataset

def store_dataset_bin(filename, dataset):

    assert dataset.ndim == 1 or dataset.ndim == 2
    assert dataset.dtype == np.float32

    num_rows, num_columns = (*dataset.shape, 1)[:2]
    with open(filename, "wb") as outfile:
        outfile.write(struct.pack("<I", num_columns))
        for i in range(num_rows):
            outfile.write(dataset[i].tobytes())

def main(model_filename, data_filename, label_filename):

    start_time = time.time()
    model = lgb.Booster(model_file=model_filename)
    end_time = time.time()
    model_load_time = end_time - start_time

    start_time = time.time()
    data_points = load_dataset_bin(data_filename)
    end_time = time.time()
    data_load_time = end_time - start_time

    start_time = time.time()
    predicted_labels = np.round(model.predict(data_points))
    end_time = time.time()
    classification_time = end_time - start_time

    start_time = time.time()
    store_dataset_bin(label_filename, predicted_labels.astype(np.float32))
    end_time = time.time()
    label_store_time = end_time - start_time

    print("Model Load Time:", model_load_time)
    print("Data Load Time:", data_load_time)
    print("Classification Time:", classification_time)
    print("Label Store Time:", label_store_time)

def parse_command_line_arguments():

    parser = argparse.ArgumentParser(description="Classify data using a pre-trained LightGBM classifier.")
    parser.add_argument("model_filename", type=pathlib.Path, metavar="MODEL_INPUT_FILE")
    parser.add_argument("data_filename", type=pathlib.Path, metavar="DATA_INPUT_FILE")
    parser.add_argument("label_filename", type=pathlib.Path, metavar="LABEL_OUTPUT_FILE")
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_command_line_arguments()
    main(**dict(vars(args)))
