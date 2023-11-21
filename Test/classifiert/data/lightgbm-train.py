#!/usr/bin/env python3

import pickle
import sys

import lightgbm as lgb

def load_dataset_bin(filename):

    import struct

    with open(filename, "rb") as inf:
        num_features, = struct.unpack("<I", inf.read(4))
        data_points, labels = [], []
        unpacker = struct.Struct("<" + "f" * (num_features + 1))
        for row in unpacker.iter_unpack(inf.read()):
            data_points.append(row[:-1])
            labels.append(row[-1])
    return data_points, labels


def main(train_filename, test_filename, nthreads=1):

    nthreads = int(nthreads)

    train_data_points, train_labels = load_dataset_bin(train_filename)
    test_data_points, test_labels = load_dataset_bin(test_filename)

    # model = lgb.LGBMClassifier(learning_rate=0.09,colsample_bytree=np.sqrt(len(data_points))/len(data_points), num_leaves=1024)
    model = lgb.LGBMClassifier(boosting_type="dart",learning_rate=0.4,num_leaves=4096)
    model.fit(train_data_points, train_labels, eval_set=[(test_data_points, test_labels),(train_data_points, train_labels)],
              eval_metric='logloss')

    # random_forest = lgb.LGBMClassifier(boosting_type="rf", max_depth=50, n_estimators=150, n_jobs=nthreads)

    # random_forest = lgb.LGBMClassifier(boosting_type="rf",
    #                          num_leaves=165,
    #                          colsample_bytree=.5,
    #                          n_estimators=150,
    #                          min_child_weight=5,
    #                          min_child_samples=10,
    #                          subsample=.632, # Standard RF bagging fraction
    #                          subsample_freq=1,
    #                          min_split_gain=0,
    #                          reg_alpha=10, # Hard L1 regularization
    #                          reg_lambda=0,
    #                          n_jobs=3)

    # print(dir(random_forest))
    # print(dir(random_forest.estimators_))
    # print(len(random_forest.estimators_))
    # print(random_forest.estimators_[0].get_depth())
    # print(dir(random_forest.estimators_[0]))
    # print(dir(random_forest.estimators_[0].tree_.node_count))
    # # print("Nodes", random_forest.tree_.node_count)

    # print("max-tree-depth", max([estimator.get_depth() for estimator in random_forest.estimators_]))
    # print("max-node-count", max([estimator.tree_.node_count for estimator in random_forest.estimators_]))

    print("train-accuracy {:.4f}".format(model.score(train_data_points, train_labels)))
    print("test-accuracy {:.4f}".format(model.score(test_data_points, test_labels)))


if __name__ == "__main__":

    main(*sys.argv[1:])
