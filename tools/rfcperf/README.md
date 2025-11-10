![Balsa Logo](../../core/Images/BalsaLogo.png)

# rfcperf: Random Forest Classifier Performance Tool

A comprehensive benchmarking and profiling tool for comparing Random Forest classifier implementations across different dataset sizes and configurations.

## Overview

rfcperf is designed to systematically evaluate and compare Random Forest classifiers by:
- Running training and classification across multiple dataset sizes
- Measuring system performance (CPU time, memory usage, wall-clock time)
- Calculating classification quality metrics (accuracy, precision, recall, etc.)
- Generating comparative visualization reports

This tool is particularly useful for:
- Benchmarking different Random Forest implementations (Balsa, scikit-learn, etc.)
- Optimizing classifier parameters for your dataset
- Understanding scalability and resource usage patterns
- Comparing trade-offs between speed, memory, and accuracy

## Features

- **Multi-format dataset support**: JSON, CSV, binary, and Balsa native formats
- **Automated dataset generation**: Creates training/test splits at multiple sizes
- **Comprehensive metrics**: System performance (time, memory) and classification quality (accuracy, F-scores, etc.)
- **Visual reports**: Multi-page PDF reports with comparison graphs
- **Caching**: Efficient dataset reuse across runs
- **Configurable**: External configuration file for classifier settings
- **Extensible**: Driver-based architecture for adding new classifiers

## Installation
```bash
cd tools
pip install -e .
```

**Requirements:**
- Python ≥ 3.7
- NumPy
- matplotlib
- jsonpickle

## Quick Start

### 1. Prepare Your Dataset

Your training data should be in JSON format (using jsonpickle):
```python
import numpy as np
import jsonpickle
import jsonpickle.ext.numpy

jsonpickle.ext.numpy.register_handlers()

# Your data: 2D array of features (float32)
data_points = np.array([[1.0, 2.0], [3.0, 4.0], ...], dtype=np.float32)

# Your labels: 1D array (float32, values 0 or 1 for binary classification)
labels = np.array([0, 1, 0, ...], dtype=np.float32)

# Save as JSON
with open("my_dataset.json", "w") as f:
    f.write(jsonpickle.encode((data_points, labels)))
```

### 2. Generate Default Configuration
```bash
python -m rfcperf profile my_dataset.json balsa
```

This creates `rfcperf.ini` with default settings. Edit this file to configure your classifiers.

### 3. Configure Classifiers

Edit `rfcperf.ini`:
```ini
[rfcperf]
cache_dir = cache
run_dir = run

[balsa]
driver = balsa
balsa_train = /path/to/balsa_train
balsa_classify = /path/to/balsa_classify
balsa_measure = /path/to/balsa_measure

[sklearn]
driver = sklearn
# sklearn-specific options...
```

### 4. Run Profiling
```bash
python -m rfcperf profile my_dataset.json balsa sklearn \
    -n 1000,5000,10000,50000 \
    -e 150 \
    -t 4 \
    -p 33
```

This profiles both Balsa and scikit-learn on datasets of 1K, 5K, 10K, and 50K samples, using 150 trees, 4 threads, and a 33% test split.

### 5. View Results

Results are saved in timestamped directories under `run/`:
```
run/2025-01-15T10:30:00/
├── balsa.pdf           # Balsa-specific report
├── sklearn.pdf         # sklearn-specific report
├── all.pdf             # Combined comparison report
├── balsa/              # Detailed logs and output
│   ├── 1000/
│   ├── 5000/
│   └── ...
└── sklearn/
    ├── 1000/
    └── ...
```

## Commands

### profile

Profile classifiers across multiple dataset sizes.
```bash
python -m rfcperf profile TRAIN_DATA_FILE CLASSIFIER [CLASSIFIER ...] [OPTIONS]
```

**Required Arguments:**
- `TRAIN_DATA_FILE`: JSON file containing training data
- `CLASSIFIER`: Name(s) of classifiers to profile (must be defined in config)

**Options:**
- `-c, --config-file FILE`: Configuration file (default: `rfcperf.ini`)
- `-n, --data-sizes SIZES`: Comma-separated list of dataset sizes (e.g., `1000,5000,10000`)
- `-p, --test-percentage PCT`: Percentage of data for testing (0-100, default: 33)
- `-T, --test-data-file FILE`: Use separate test dataset file (mutually exclusive with `-p`)
- `-e, --num-estimators N`: Number of trees in forest (default: 150)
- `-d, --max-tree-depth N`: Maximum tree depth (default: 50)
- `-f, --num-features N`: Number of features to consider at each split
- `-t, --num-threads N`: Number of threads to use (default: 1)
- `-s, --random-seed N`: Random seed for reproducibility
- `-x, --timeout N`: Timeout in seconds for each run
- `-C, --no-cache`: Force regeneration of cached datasets

**Example:**
```bash
# Profile Balsa with different thread counts
python -m rfcperf profile data.json balsa \
    -n 10000,50000,100000 \
    -e 200 \
    -d 30 \
    -t 8 \
    -s 42

# Compare multiple classifiers
python -m rfcperf profile data.json balsa sklearn ranger \
    -n 5000,10000,20000 \
    -p 25 \
    -e 150
```

### sample

Extract a sample from an existing dataset and save in various formats.
```bash
python -m rfcperf sample DATA_INPUT_FILE DATA_OUTPUT_FILE [LABEL_OUTPUT_FILE] [OPTIONS]
```

**Required Arguments:**
- `DATA_INPUT_FILE`: Input JSON dataset
- `DATA_OUTPUT_FILE`: Output data file
- `LABEL_OUTPUT_FILE`: Output label file (not needed for CSV format)

**Options:**
- `-f, --data-format FORMAT`: Output format (`csv`, `bin`, `balsa`, default: `bin`)
- `-n, --sample-size N`: Number of samples to draw (default: use all)
- `-r, --with-replacement`: Sample with replacement (allows duplicates)
- `-s, --random-seed N`: Random seed for reproducibility

**Examples:**
```bash
# Convert JSON to Balsa format
python -m rfcperf sample data.json data.balsa labels.balsa -f balsa

# Draw 1000 random samples to CSV
python -m rfcperf sample data.json sample.csv -f csv -n 1000 -s 42

# Oversample with replacement
python -m rfcperf sample small_data.json large_data.bin large_labels.bin \
    -n 100000 -r
```

## Configuration File

The configuration file (`rfcperf.ini`) uses INI format:
```ini
[rfcperf]
cache_dir = cache          # Directory for cached datasets
run_dir = run              # Directory for output results

[classifier_name]
driver = driver_type       # Driver to use (e.g., balsa, sklearn)
# Driver-specific options below
option1 = value1
option2 = value2
```

Each classifier section defines:
- **driver**: Which driver implementation to use
- **Driver-specific parameters**: Paths to executables, additional options

## Performance Metrics

rfcperf measures and reports:

### System Performance
- **Wall-clock time**: Total elapsed time
- **CPU time**: User + system CPU time
- **CPU utilization**: Percentage of available CPU used
- **Peak memory (RSS)**: Maximum resident set size
- **Timing breakdown**: Data loading, training, classification, I/O times
- **Model properties**: Tree depth, node count

### Classification Quality
- **Accuracy**: Overall correctness
- **TPR/TNR**: True positive/negative rates (sensitivity/specificity)
- **PPV/NPV**: Positive/negative predictive values (precision)
- **P4 metric**: Harmonic mean of TPR, TNR, PPV, NPV
- **DOR**: Diagnostic odds ratio
- **Confusion matrix**: Full classification breakdown

See [core/README.md](../core/README.md#performancemetrics) for detailed metric definitions.

## Output Structure

Each profiling run creates a timestamped directory:
```
run/TIMESTAMP/
├── classifier1.pdf              # Individual classifier report
├── classifier2.pdf
├── all.pdf                      # Combined comparison report
├── classifier1/
│   ├── 1000/                    # Results for 1000 samples
│   │   ├── model.balsa          # Trained model
│   │   ├── predictions.balsa    # Classification output
│   │   ├── train-time.txt       # GNU time output
│   │   ├── test-time.txt
│   │   ├── train-stdout.txt     # Program output
│   │   ├── train-stderr.txt
│   │   ├── test-stdout.txt
│   │   └── test-stderr.txt
│   ├── 5000/
│   └── ...
└── classifier2/
    └── ...
```

## Report Contents

Generated PDF reports include graphs showing:

1. **Training performance**:
   - Wall-clock time vs. dataset size
   - CPU time and utilization
   - Memory usage (peak RSS)
   - Data loading time
   - Model storage time
   - Tree statistics (depth, node count)

2. **Classification performance**:
   - Wall-clock time vs. dataset size
   - CPU time and utilization
   - Memory usage
   - Model loading time
   - Classification time

3. **Classification quality**:
   - Accuracy across dataset sizes
   - Precision (PPV), recall (TPR)
   - Specificity (TNR), NPV
   - P4 metric
   - Diagnostic odds ratio

## Caching

rfcperf caches generated datasets in the `cache/` directory to avoid regeneration:
```
cache/
├── train-data-1000-oob-33.balsa
├── train-label-1000-oob-33.balsa
├── test-data-1000-oob-33.balsa
├── test-label-1000-oob-33.balsa
├── train-data-1000-oob-33.bin
├── train-label-1000-oob-33.bin
└── ...
```

Cached datasets are reused across runs with the same parameters unless:
- `--no-cache` flag is used
- A random seed is specified (forces regeneration)
- The cache directory is manually cleared

## Extending rfcperf

To add support for new classifiers:

1. Create a new driver in the `drivers/` module
2. Implement the driver interface (train, classify, measure)
3. Add configuration section to `rfcperf.ini`
4. Register the driver in `drivers/__init__.py`

See existing drivers for implementation examples.

## Support

Support queries may be directed at info@jigsaw.nl
