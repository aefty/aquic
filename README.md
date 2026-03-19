# [!!!!AI TEXT!!!!]AQUIC — XXXXXX Inverse Covariance Estimation

 statistical toolkit for estimating sparse inverse covariance (precision) matrices using adaptively chosen quantile parameters. Includes benchmarking against classical methods and applications to fMRI brain imaging and gene expression data.

---

## Requirements

### Python

**Python 3.12** (the compiled C++ extension `quic_pybind` targets CPython 3.12)

### Python Libraries

| Package | Version | Purpose |
|---|---|---|
| numpy | 1.26.4 | Numerical arrays and linear algebra |
| scipy | 1.17.1 | Scientific computing, special functions |
| scikit-learn | 1.8.0 | GraphicalLasso, cross-validation, metrics |
| joblib | 1.5.3 | Parallel cross-validation folds |
| pandas | 3.0.1 | Data loading and manipulation |
| networkx | 3.6.1 | Graph-based covariance structure analysis |
| gglasso | 0.2.1 | Group graphical lasso baseline |
| rpy2 | 3.6.5 | R integration (TIGER, CLIME, `huge` data generation) |
| nibabel | 5.4.0 | Loading fMRI `.pconn.nii` neuroimaging files |
| pybind11 | 3.0.2 | Build tool for the C++/Python extension |
| metis | 0.2a5 | Graph partitioning tool |
| tqdm  | 4.67.3 | Loading bars for longer experiments |
| matplotlib | 3.10.8 | Matrix plotting, results plotting |
| plotly | 6.6.0 | Brain parcel classification plotting |
| nbformat | 5.10.4 | Tools for brain parcel classification plotting |

### R Libraries (via rpy2)

- `huge` — synthetic graph/covariance data generation
- `flare` — Version 1.7.0, implements TIGER and CLIME estimators ([CRAN](https://cran.r-project.org/web/packages/flare/flare.pdf))
- `MASS` - Support Functions and Datasets for Venables and Ripley's MASS ([CRAN](https://cran.r-project.org/web/packages/MASS/index.html))

### System / C++ Dependencies (for building `quic_pybind`)

| Dependency | Source | Purpose |
|---|---|---|
| g++ (C++23) | system | Compiler |
| Eigen 3 | `brew install eigen` | Matrix algebra in C++ |
| Boost | `brew install boost` | `boost::math::erfinv` |
| libomp | `brew install libomp` | OpenMP parallelism |
| LAPACK + BLAS | macOS Accelerate / system | Dense linear algebra |
| Python 3.12 dev headers | python3.12-config | pybind11 linking |

---

## Installation

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Install system libraries (macOS / Homebrew)
brew install eigen boost libomp

# 3. Build the C++ extension
cd aquic
make
cp quic_pybind*.so ../   # copy to project root
```


---

## Project Structure

```
aquic/
├── aquic/
│   ├── QUIC.cpp                   # Core QUIC algorithm (GPL v3)
│   ├── quic_pybind.cpp            # pybind11 Python/C++ bindings (AQUIC, QUIC)
│   └── makefile                   # Build configuration
├── data/
│   ├── fMRI/                      # Brain parcellation + covariance matrix
│   └── gene/                      # Gene expression CSV data
├── results/                       # Cached results (pickle files)
├── _util.py                       # Estimator wrappers and evaluation metrics
├── _QUICGraphicalLassoCV.py       # Sklearn-compatible QUIC with CV
├── test.ipynb                     # Synthetic benchmark experiments
├── test_app_brain.ipynb           # fMRI application demo
├── test_app_gene.ipynb            # Gene expression application demo
├── test_verify_quic_cv.ipynb      # Cross-validation verification
└── requirements.txt
```

---

## Methods

### AQUIC (this work)

- `compute_aquic` — Adaptive Quantile Inverse Covariance (standard)

### Baselines

- `compute_glasso_cv` — Graphical Lasso with cross-validated `rho` (scikit-learn)
- `compute_quic_cv` — QUIC with cross-validated `alpha`
- `compute_ledoit` — Ledoit-Wolf shrinkage estimator
- `compute_tiger` — TIGER (via R `flare`)
- `compute_clime` — CLIME (via R `flare`)
- `compute_scov` — Sample covariance

---

## Applications

- **Synthetic benchmarks** — random, hub, cluster, band, scale-free graph structures
- **fMRI** — Cole-Anticevic brain parcellation network (cortex + subcortex)
- **Gene expression** — sparse regulatory network recovery
