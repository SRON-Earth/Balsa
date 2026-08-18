![](core/Images/BalsaLogo.png)

# Balsa: A Fast Random Forest Classifier

[![DOI](https://joss.theoj.org/papers/10.21105/joss.08778/status.svg)](https://doi.org/10.21105/joss.08778)

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
On Linux/WSL, after installing libraries, the dynamic linker cache must be updated. 
```bash
sudo ldconfig
```
This is step is not needed for macOSX.
See [core/README.md](core/README.md#installation) for details (e.g. using a custom installation path).

### Running the unit tests
After building Balsa, you can verify the installation by running the unit tests:

```bash
balsa_test
```

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

## Citation

If you use Balsa in your research, please cite the JOSS paper:

```bibtex
@article{Borsdorff2026,
  doi = {10.21105/joss.08778},
  url = {https://doi.org/10.21105/joss.08778},
  year = {2026},
  publisher = {The Open Journal},
  author = {Tobias Borsdorff and Denis de Leeuw Duarte and Joris van Zwieten and Soumyajit Mandal and Jochen Landgraf},
  title = {Balsa: A Fast C++ Random Forest Classifier with Command-line and Python Interface},
  journal = {Journal of Open Source Software}
}
```

## Support

Developed for [SRON Netherlands Institute for Space Research](https://www.sron.nl) by [Jigsaw B.V.](https://www.jigsaw.nl) with funding from [ESA](https://www.esa.int).

For support: info@jigsaw.nl
