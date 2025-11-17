![](core/Images/BalsaLogo.png)

# Balsa: A Fast Random Forest Classifier

Balsa is a fast and memory-efficient implementation of the Random Forest classification algorithm, optimized for low memory usage and high speed during both training and classification.

## Repository Structure

This repository contains three main components:

| Component | Description | Documentation |
|-----------|-------------|---------------|
| **[core/](core/)** | C++ implementation and command-line tools | [core/README.md](core/README.md) |
| **[python/](python/)** | Python bindings for Balsa | [python/README.md](python/README.md) |
| **[tools/](tools/)** | Performance analysis utilities | [tools/README.md](tools/README.md) |

## Where to Start

**→ New to Balsa?** Start with **[core/README.md](core/README.md)** for complete theory, installation, and tutorials.

**→ Want command-line tools?** See **[core/README.md](core/README.md#balsacommandline)** for `balsa_train`, `balsa_classify`, `balsa_measure`, etc.

**→ Using Python?** Go to **[python/README.md](python/README.md)** for Python API and pip installation.

**→ Need C++ API?** Check **[core/README.md](core/README.md#usingbalsacpp)** for C++ integration.

**→ Analyzing models?** See **[tools/README.md](tools/README.md)** for performance evaluation tools.

**→ Optimizing performance?** Read **[core/README.md](core/README.md#optimizingsystemperformance)** for tuning guidelines.

## Quick Start

### Command-Line
```bash
cd core && mkdir build && cd build
cmake .. && make && sudo make install
```
See [core/README.md](core/README.md#installation) for details (e.g. using a custom installation path).

### Python
```bash
cd python && pip install .
```
See [python/README.md](python/README.md#installation) for details.

## Key Features

- High-performance C++ core with multi-threading support
- Command-line tools for training, classification, and analysis
- Python bindings with NumPy integration
- Comprehensive performance metrics (accuracy, precision, recall, F-score, etc.)
- Feature importance analysis
- Efficient handling of large datasets

## Support

Developed for [SRON Netherlands Institute for Space Research](https://www.sron.nl) by [Jigsaw B.V.](https://www.jigsaw.nl) with funding from [ESA](https://www.esa.int).

For support: info@jigsaw.nl
