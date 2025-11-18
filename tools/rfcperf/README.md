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

**Requirements:**
- Python ≥ 3.7
- NumPy
- matplotlib
- jsonpickle
- GNU time command


## Quick Start

### 1. Prepare Your Dataset

#### Create Dataset Specification

First, create a dataset specification file named `tropomi.conf`:
```
multisource(7)
{
    source(50)
    {
        gaussian(0.16596016, 0.16781704);
        gaussian(30.953608, 18.591196);
        gaussian(12.02204, 46.754826);
        gaussian(91839.86, 11728.834);
        gaussian(0.5906968, 0.089086026);
        gaussian(0.6995243, 0.19331227);
        gaussian(0.53668684, 0.0888503);
}
    source(50)
    {
        gaussian(0.22127023, 0.19662453);
        gaussian(26.177174, 16.360184);
        gaussian(0.29663125, 49.04709);
        gaussian(90820.484, 11894.523);
        gaussian(0.606064, 0.0975867);
        gaussian(0.86118674, 0.07939516);
        gaussian(0.56880575, 0.09053809);
}
}
```

This specification defines a binary classification problem with 7 features per data point. The two sources represent different classes (e.g., cloudy and clear-sky satellite measurements of TROPOMI) with a 50:50 distribution ratio.

#### Generate Dataset Using balsa_generate

Use the `balsa_generate` tool to create structured random test data:
```bash
# Generate 3 million data points based on tropomi.conf specification
balsa_generate -p 3000000 tropomi.conf tropomi-points.balsa tropomi-labels.balsa

# Convert binary format to readable text files
balsa_print tropomi-points.balsa > fpoints.txt
balsa_print tropomi-labels.balsa > flabels.txt
```

#### Convert to JSON Format

Once you have the text files, convert them to JSON format for use with the Python bindings:
```python
import numpy as np
import jsonpickle

# Paths to the input text files
points_file = "fpoints.txt"
labels_file = "flabels.txt"
output_json = "my_dataset.json"

def read_points(filename):
    """Read floating point table from text file."""
    data = []
    with open(filename, "r") as f:
        lines = f.readlines()
    # Skip header lines until we find table rows
    for line in lines:
        line = line.strip()
        if not line or line[0].isdigit() == False:
            continue
        # Row format: "0   : 123.303  43.3323  -5.20379 102.836"
        parts = line.split(":")
        values = parts[1].strip().split()
        data.append([float(v) for v in values])
    return np.array(data, dtype=np.float32)

def read_labels(filename):
    """Read label table from text file."""
    labels = []
    with open(filename, "r") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if not line or line[0].isdigit() == False:
            continue
        # Row format: "0   : 0"
        parts = line.split(":")
        labels.append(int(parts[1].strip()))
    return np.array(labels, dtype=np.float64)

# Load data
data_points = read_points(points_file)
labels = read_labels(labels_file)

# Make sure shapes match
assert len(data_points) == len(labels), "Number of points and labels must match"

# Encode and write JSON
with open(output_json, "w") as f:
    f.write(jsonpickle.encode([data_points, labels]))

print(f"Dataset written to {output_json}")
print(f"Data shape: {data_points.shape}")
print(f"Labels shape: {labels.shape}")
```

### 2. Generate Default Configuration
```bash
python -m rfcperf profile my_dataset.json balsa
```

This creates `rfcperf.ini` with default settings. Edit this file to configure your classifiers. You need to install sklearn in your python environment and download and built ranger a cpp implementation of the random forest classifier concept (https://github.com/imbs-hl/ranger).

### 3. Configure Classifiers

Edit `rfcperf.ini`:
```ini
[rfcperf]
cache_dir = cache
run_dir = run

[balsa]
driver = balsa
path = /path/to/dir/that/contains/balsa/binaries

[ranger]
driver = ranger
path = /path/to/dir/that/contains/ranger/binary

[sklearn]
driver = sklearn
python = /path/to/python/interpreter

```

### 4. Run Profiling
```bash
mkdir run cache
python -m rfcperf profile my_dataset.json balsa sklearn ranger \
    -n 1000,250000,500000,750000,1000000 \
    -e 150 \
    -t 8 \
    -p 33 \
    -f 2 \
    -d 50
```

This profiles Balsa, scikit-learn and the ranger library on datasets of 1K, 2.5M, 5M, 7.5M, and 10M samples, using 150 trees, 16 threads, and a 33% test split. This will take some time if you want to speed it up reduce the dataset size.

### 5. View Results

Results are saved in timestamped directories under `run/`:
```
run/2025-01-15T10:30:00/
├── balsa.pdf           # Balsa-specific report
├── sklearn.pdf         # sklearn-specific report
├── all.pdf             # Combined comparison report
├── balsa/              # Detailed logs and output
│   ├── 1000/
│   ├── 2500000/
│   └── ...
└── sklearn/
    ├── 2500000/
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
python -m rfcperf sample my_dataset.json data.balsa labels.balsa -f balsa

# Draw 1000 random samples to CSV
python -m rfcperf sample my_dataset.json sample.csv -f csv -n 1000 -s 42

# Oversample with replacement
python -m rfcperf sample my_dataset.json large_data.bin large_labels.bin \
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

See [core/README.md](../../core/README.md) for detailed metric definitions.

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
