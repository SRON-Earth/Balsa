![](images/BalsaLogo.png)

# Balsa: A Fast Random Forest Classifier

## About

Balsa is a fast and memory-efficient implementation of the RandomForest
classification algorithm. Balsa is optimized for low memory usage and high
speed during both training and classification. It is particularly useful for
training on larger datasets, and for near-real time classification.

This package contains the Python bindings for Balsa. The C++ implementation of
Balsa contains extensive documentation, covering performance- and
classification-optimization. This file covers the specifics of using the Python
binding.

## Prerequisites

The balsa-python package uses NumPy.

## Installation

Balsa can be installed directly from a cloned Git repository, as follows:

	pip install .

Installation of the balsa-python package using `pip` requires a C/C++ compiler,
CMake (>=3.18), a Python interpreter, NumPy, and the Python development headers
and static library (on Ubuntu the required package is called `python3-dev`).

## Examples

### Training and Classification with Default Settings

The following example code demonstrates how the Balsa trainer and classifier are
called from Python, using all default parameters:

	import numpy as np
	import balsa

	# Create a data set with two features per point.
	dataset = np.array(((1, 1), (1, -1), (-1, -1), (-1, 1), (0, 0)),dtype=np.float64)
	
	# Create labels.
	labels = np.array((0, 1, 0, 1, 1), dtype=np.uint8)

	# Train a model, write the output to 'model.balsa'. 
	balsa.train(dataset, labels, "model.balsa" )
	
	# Create a classifier (for 2 features).
	classifier = balsa.RandomForestClassifier("model.balsa", 2)
	
	# Classify the dataset.
	predictions = classifier.classify(dataset)

### Additional Training Options

The trainer has several options that can be accessed through the constructor and
`train()` method:

	balsa.train( dataset, labels, "model.balsa", 
	             features_to_scan = 2, 
	             max_depth = 10,
	             tree_count = 1,
	             concurrent_trainers = 1, 
	             single_precision = False )
	

The `single_precision` flag switches from double-precision representation to
single-precision. This leads to reduced memory usage and faster training, at
the expense of precision, and possibly predictive performance. The purpose of
the other parameters is described in the C++ Balsa documentation.

### Additional Classification Options

Options for classification are passed to the classifier constructor:

	classifier = balsa.RandomForestClassifier( "model.balsa", 2, 
	             max_threads=1,
	             max_preload=0, 
	             single_precision = False )

The `single_precision` flag switches from double-precision representation to
single-precision. N.B. using a different representation during training and
classification incurs a performance penalty during model-loading. The purpose
of the other parameters is described in the C++ Balsa documentation.
