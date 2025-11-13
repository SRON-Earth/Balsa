![](images/BalsaLogo.png)

# Balsa: A Fast Random Forest Classifier

## About

Balsa is a fast and memory-efficient implementation of the random forest
classification algorithm. Balsa is optimized for low memory usage and high
speed during both training and classification. It is particularly useful for
training on larger datasets, and for near-real time classification.

This package contains the Python bindings for Balsa. The C++ implementation of
Balsa contains extensive documentation, covering performance- and
classification-optimization. This file covers the specifics of using the Python
binding.

## Prerequisites

The balsa-python package requires NumPy >= 1.25.

## Installation

Balsa can be installed directly from a cloned Git repository, as follows:

```
pip install .
```

Installation of the balsa Python package using `pip` requires a working C/C++
compiler, git, CMake (>=3.18), Python (>=3.9), NumPy (>= 1.25), and the Python
development headers and static library (on Ubuntu the required package is
called `python3-dev`).

## Examples

### Training and Classification with Default Settings

The following example code demonstrates how the Balsa trainer and classifier can
be called from Python, using all default parameters:

```
import numpy as np
import balsa_rfc

# Create a data set with two features per point.
dataset = np.array(((1, 1), (1, -1), (-1, -1), (-1, 1), (0, 0)),dtype=np.float64)

# Create labels.
labels = np.array((0, 1, 0, 1, 1), dtype=np.uint8)

# Train a model, write the output to 'model.balsa'.
balsa_rfc.train(dataset, labels, "model.balsa")

# Create a classifier from the trained random forest.
classifier = balsa_rfc.RandomForestClassifier("model.balsa")

# Classify the dataset.
predictions = classifier.classify(dataset)
```

### Additional Training Options

The trainer has several options that can be accessed through the constructor and
`train()` method:

```
balsa_rfc.train(dataset, labels, "model.balsa",
            features_to_consider = 0,
            max_depth = 4294967295,
            min_purity = 1.0,
            tree_count = 150,
            concurrent_trainers = 1)
```

The purpose of these parameters is described in the docstring of `balsa.train`,
as well as in the Balsa C++ documentation.

The dataset will be cast to single-precision floats (np.float32) if this is
possible without causing rounding or truncation. This leads to reduced memory
usage and faster training, at the expense of precision, and possibly predictive
performance.

### Additional Classification Options

Options for classification are passed to the classifier constructor:

```
classifier = balsa_rfc.RandomForestClassifier("model.balsa",
                                          max_threads = 0,
                                          max_preload = 1)
```

The purpose of these parameters is described in the docstring of
`RandomForestClassifier`, as well as in the Balsa C++ documentation.
