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

def load_lgb_dataset(data_filename, label_filename):

    data = load_dataset_bin(data_filename)
    label = load_dataset_bin(label_filename)
    num_rows, num_columns = data.shape
    assert label.shape == (num_rows, 1)
    label.shape = (-1,)
    return lgb.Dataset(data, label == 1.0), num_columns

def main(data_filename, label_filename, model_filename, num_estimators, random_seed, max_tree_depth, num_features, num_threads):

    start_time = time.time()
    train_set, num_data_features = load_lgb_dataset(data_filename, label_filename)
    end_time = time.time()
    data_load_time = end_time - start_time

    # Feature fraction is an unfortunate parameterization. It is unclear how the
    # number of features to consider is computed internally. Rounding or
    # truncation rouding could cause more or less features to be used than
    # intended.
    if num_features is None:
        num_features = int(np.sqrt(num_data_features))
    feature_fraction_bynode = num_features / num_data_features

    params = {
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": "binary_logloss",
        "num_trees": num_estimators,
        "learning_rate": 0.4,
        "num_leaves": 4096,
        "tree_learner": "serial",
        "num_threads": num_threads,
        "max_depth": max_tree_depth,
        "min_data_in_leaf": 1,
        "feature_fraction_bynode": feature_fraction_bynode
    }

    if random_seed is not None:
        params["seed"] = random_seed
        params["deterministic"] = True
        params["force_row_wise"] = True

    start_time = time.time()
    model = lgb.train(params, train_set)
    end_time = time.time()
    training_time = end_time - start_time

    start_time = time.time()
    model.save_model(model_filename)
    end_time = time.time()
    model_store_time = end_time - start_time

    print("Data Load Time:", data_load_time)
    print("Training Time:", training_time)
    print("Model Store Time:", model_store_time)

def parse_command_line_arguments():

    def positive_integer(text):
        value = int(text)
        if value <= 0:
            raise ValueError
        return value

    parser = argparse.ArgumentParser(description="Train a LightGBM classifier.")
    parser.add_argument("data_filename", type=pathlib.Path, metavar="DATA_INPUT_FILE")
    parser.add_argument("label_filename", type=pathlib.Path, metavar="LABEL_INPUT_FILE")
    parser.add_argument("model_filename", type=pathlib.Path, metavar="MODEL_OUTPUT_FILE")
    parser.add_argument("-d", "--max-tree-depth", type=positive_integer)
    parser.add_argument("-f", "--num-features", type=positive_integer)
    parser.add_argument("-e", "--num-estimators", type=positive_integer, default="150")
    parser.add_argument("-t", "--num-threads", type=positive_integer, default="1")
    parser.add_argument("-s", "--random-seed", type=positive_integer)
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_command_line_arguments()
    main(**dict(vars(args)))
