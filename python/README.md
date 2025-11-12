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
```bash
pip install .
```
Installation of the balsa Python package using `pip` requires a working C/C++
compiler, git, CMake (>=3.18), Python (>=3.9), NumPy (>= 1.25), and the Python
development headers and static library (on Ubuntu the required package is
called `python3-dev`).

## Quick Start

### Basic Training and Classification
The following example demonstrates training and classification with default settings:

```python
import numpy as np
import balsa

# Create a dataset with two features per point
dataset = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1], [0, 0]], dtype=np.float64)
labels = np.array([0, 1, 0, 1, 1], dtype=np.uint8)

# Train a model and save it to 'model.balsa'
balsa.train(dataset, labels, "model.balsa")

# Load the trained model
classifier = balsa.RandomForestClassifier("model.balsa")

# Classify the dataset
predictions = classifier.classify(dataset)
print(f"Predictions: {predictions}")
```

## Training Options

### Basic Training Parameters
```python
balsa.train(
    dataset,                      # Training data: shape [n_samples, n_features]
    labels,                    # Ground truth labels: shape [n_samples]
    "model.balsa",            # Path to save the trained model
    features_to_consider=0,    # Features per split (0 = sqrt of total)
    max_depth=4294967295,      # Maximum tree depth
    min_purity=1.0,           # Minimum Gini-purity (1.0 = pure nodes)
    tree_count=150,           # Number of trees in the forest
    concurrent_trainers=1      # Number of parallel training threads
)
```

### Parameter Guide

**`features_to_consider`**: Number of features randomly selected when splitting a node
- `0` (default): Uses sqrt(total_features)
- Higher values: More thorough splits, longer training time
- Lower values: Faster training, more randomness

**`max_depth`**: Maximum distance from any node to tree root
- Default is effectively unlimited
- Lower values: Prevent overfitting, faster training and classification
- Higher values: May capture more complex patterns but risk overfitting

**`min_purity`**: Minimum Gini-purity threshold for splitting
- `1.0` (default): Split until all samples in a node have the same label
- `1/n_classes`: No splitting at all
- Values between: Earlier stopping, faster training, regularization

**`tree_count`**: Number of decision trees in the ensemble
- `150` (default): Good balance for most tasks
- More trees: Better predictions, longer training/classification
- Fewer trees: Faster but may underfit

**`concurrent_trainers`**: Parallel training threads
- `1` (default): Single-threaded training
- Higher values: Faster training on multi-core systems

### Example: Custom Training for Large Dataset
```python
# Optimize for speed and memory on large dataset
balsa.train(
    dataset, 
    labels, 
    "model.balsa",
    features_to_consider=10,   # Consider 10 features per split
    max_depth=2,              # Limit tree depth
    min_purity=0.95,          # Allow slightly impure nodes
    tree_count=100,           # Fewer trees for speed
    concurrent_trainers=8      # Use 8 cores
)
```

## Classification Options

### Creating a Classifier
```python
classifier = balsa.RandomForestClassifier(
    model_filename,     # Path to trained model file
    max_threads=0,      # Worker threads (0 = single-threaded)
    max_preload=1       # Number of trees to preload
)
```

### Parameter Guide

**`max_threads`**: Number of worker threads for classification
- `0` (default): Single-threaded classification
- `> 0`: Multi-threaded classification for faster inference

**`max_preload`**: Number of decision trees to preload into memory
- `0`: Load entire forest into memory (fastest classification, highest memory)
- `1` (default): Load one tree at a time (lowest memory, slower)
- `n * max_threads`: Good compromise for multi-threaded classification

### Memory Management Examples

```python
# Maximum speed: Load entire forest
classifier_fast = balsa.RandomForestClassifier(
    "model.balsa",
    max_threads=8,
    max_preload=0  # Load all trees
)

# Minimum memory: Stream trees
classifier_memory_efficient = balsa.RandomForestClassifier(
    "model.balsa",
    max_threads=1,
    max_preload=1  # Load one tree at a time
)

# Balanced: Multi-threaded with reasonable memory
classifier_balanced = balsa.RandomForestClassifier(
    "model.balsa",
    max_threads=4,
    max_preload=12  # 3x threads
)
```

## Classifier Methods

### Get Model Information
```python
# Get number of classes the model can predict
n_classes = classifier.get_class_count()

# Get number of features expected by the model
n_features = classifier.get_feature_count()
```

### Set Class Weights
Adjust the relative importance of each class during classification:

```python
# Equal weights (default behavior)
classifier.set_class_weights(np.array([1.0, 1.0], dtype=np.float32))

# Emphasize class 1 (e.g., for imbalanced datasets)
classifier.set_class_weights(np.array([1.0, 2.0], dtype=np.float32))

# Ignore class 0 predictions
classifier.set_class_weights(np.array([0.0, 1.0], dtype=np.float32))
```

**Note**: All weights must be non-negative, and the number of weights must match the number of classes.

### Classify Data
```python
# Classify a single sample
single_sample = np.array([[1.5, -0.5]], dtype=np.float32)
prediction = classifier.classify(single_sample)

# Classify multiple samples
test_data = np.array([[1, 1], [0, 0], [-1, -1]], dtype=np.float32)
predictions = classifier.classify(test_data)
```

## Data Type Support

Balsa accepts both 32-bit and 64-bit floating-point data:

```python

# Single precision (recommended for speed/memory)
labels = np.array([0, 1], dtype=np.uint8)
data_float32 = np.array([[1, 2], [3, 4]], dtype=np.float32)
balsa.train(data_float32, labels, "model32.balsa")

# Double precision (for higher numerical accuracy)
labels = np.array([0, 1], dtype=np.uint8)
data_float64 = np.array([[1, 2], [3, 4]], dtype=np.float64)
balsa.train(data_float64, labels, "model64.balsa")
```

**Note**: Data will be automatically cast to `float32` if possible without loss of precision, reducing memory usage and improving speed.

## Complete Workflow Example

```python
import numpy as np
import balsa

# Load or create your dataset
train_data = np.random.randn(1000, 10).astype(np.float32)
train_labels = np.random.randint(0, 3, size=1000, dtype=np.uint8)
test_data = np.random.randn(200, 10).astype(np.float32)

# Train the model
print("Training random forest...")
balsa.train(
    train_data,
    train_labels,
    "my_model.balsa",
    tree_count=100,
    concurrent_trainers=4
)

# Load and inspect the model
print("Loading classifier...")
classifier = balsa.RandomForestClassifier("my_model.balsa", max_threads=4)
print(f"Model classifies {classifier.get_class_count()} classes")
print(f"Model expects {classifier.get_feature_count()} features")

# Optional: Adjust class weights
weights = np.array([1.0, 1.5, 2.0], dtype=np.float32)
classifier.set_class_weights(weights)

# Classify test data
print("Classifying test data...")
predictions = classifier.classify(test_data)
print(f"Predictions shape: {predictions.shape}")
print(f"Unique predictions: {np.unique(predictions)}")
```

## Further Documentation

For detailed information about the algorithm, performance optimization, and C++ API, please refer to the Balsa C++ documentation.  
