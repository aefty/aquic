# The Regularization Parameter: Sparse Precision Matrix Estimation

 Statistical toolkit for estimating sparse inverse covariance (precision) matrices using a novel, data adaptive regularization parameter. Includes benchmarking against classical methods and applications to fMRI brain imaging and gene expression data.

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

- `huge` — Version 1.5, synthetic graph/covariance data generation ([CRAN](https://cran.r-project.org/web/packages/huge))
- `flare` — Version 1.7.0, implements TIGER and CLIME estimators ([CRAN](https://cran.r-project.org/web/packages/flare))
- `MASS` - Version 7.3.65, Support Functions and Datasets for Venables and Ripley's MASS ([CRAN](https://cran.r-project.org/web/packages/MASS))

### System / C++ Dependencies (for building `quic_pybind`)

| Dependency | Source | Purpose |
|---|---|---|
| g++ (C++23) | system | Compiler |
| Eigen 3 | `brew install eigen` | Matrix algebra in C++ |
| Boost | `brew install boost` | `boost::math::erfinv` |
| libomp | `brew install libomp` | OpenMP parallelism (not used in the current implementation) |
| LAPACK + BLAS | macOS Accelerate / system | Dense linear algebra |
| Python 3.12 dev headers | python3.12-config | pybind11 linking |

---

## Installation

> **macOS**: a precompiled shared library (`quic_pybind.cpython-312-darwin.so`) is included and can be used directly — no build step required.
>
> **All other platforms**: the `makefile` in `source/` must be edited and compiled for your specific platform before use.

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. (Non-macOS only) Install system libraries and build the C++ extension
brew install eigen boost libomp   # adjust for your package manager
cd source
make
make install  # copies the shared library to project root
```

---

## Project Structure

```
aquic/
├── source/
│   ├── QUIC.cpp                   # Core QUIC algorithm (GPL v3)
│   ├── quic_pybind.cpp            # pybind11 Python/C++ bindings (AQUIC, QUIC)
│   └── makefile                   # Build configuration
├── data/
│   ├── fMRI/                      # Cole-Anticevic brain parcellation data matrix
│   └── gene/                      # Gene expression CSV data (HapMap LCL)
├── results/                       # Cached results (pickle files)
├── _util.py                       # Estimator wrappers and evaluation metrics
├── _QUICGraphicalLassoCV.py       # Sklearn-compatible QUIC with CV
├── test_app_brain.ipynb           # fMRI application demo
├── test_app_gene.ipynb            # Gene expression application demo
├── test_verify_quic_cv.ipynb      # Cross-validation verification
└── requirements.txt
```

---

## Methods

### AQUIC (this work)

- `compute_aquic` — QUIC algorithem coupled with proposed regularization parameter.

### Compariable Packages

- `compute_glasso_cv` — Graphical Lasso with cross-validated `rho` (scikit-learn)
- `compute_quic_cv` — QUIC with cross-validated `alpha`
- `compute_tiger` — TIGER (via R `flare`)
- `compute_clime` — CLIME (via R `flare`)

---

## Datasets

- **Synthetic benchmarks** — random, hub, cluster, band, scale-free graph structures using `huge` package in R.
- **fMRI data** — Ji et al. (2019), [Mapping the human brain's cortical-subcortical functional network organization](https://www.sciencedirect.com/science/article/pii/S1053811918319657), NeuroImage.
- **Gene expression data** — Mohammadi & Wit (2015), [Bayesian Structure Learning in Sparse Gaussian Graphical Models](https://projecteuclid.org/journals/bayesian-analysis/volume-10/issue-1/Bayesian-Structure-Learning-in-Sparse-Gaussian-Graphical-Models/10.1214/14-BA889.full), Bayesian Analysis.

---

## Citations

If you use this code for your research, please cite:

```bibtex
% TODO: add citation for the AQUIC paper once published
@article{AQUIC_CITATION,
  title   = {},
  author  = {},
  journal = {},
  year    = {},
  doi     = {}
}
```

### References

- **Graphical Lasso** — Friedman, Hastie & Tibshirani (2008), *Sparse inverse covariance estimation with the graphical lasso*, Biostatistics. [https://doi.org/10.1093/biostatistics/kxm045](https://doi.org/10.1093/biostatistics/kxm045)
- **QUIC / QUIC-CV** — Hsieh et al. (2011), *Sparse Inverse Covariance Matrix Estimation Using Quadratic Approximation*, NeurIPS. [https://proceedings.neurips.cc/paper/2011/file/2ba8698b79439589fdd2b0f7218d8b07-Paper.pdf](https://proceedings.neurips.cc/paper/2011/file/2ba8698b79439589fdd2b0f7218d8b07-Paper.pdf)
- **TIGER** — Liu & Zhao (2017), *Tiger: A tuning-insensitive approach for optimally estimating Gaussian graphical models*, Electronic Journal of Statistics. [https://doi.org/10.1214/16-EJS1195](https://doi.org/10.1214/16-EJS1195)
- **CLIME** — Cai, Liu & Luo (2011), *A constrained ℓ₁ minimization approach to sparse precision matrix estimation*, JASA. [https://doi.org/10.1198/jasa.2011.tm10155](https://doi.org/10.1198/jasa.2011.tm10155)
