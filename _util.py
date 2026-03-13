import quic_pybind

from rpy2.robjects import numpy2ri
import rpy2.robjects as robjects
from rpy2.robjects.conversion import localconverter

from gglasso.problem import glasso_problem
from types import SimpleNamespace
from sklearn.covariance import LedoitWolf
from sklearn.covariance import GraphicalLassoCV, GraphicalLasso
from sklearn.metrics import precision_score, recall_score, f1_score

import networkx as nx
import pandas as pd
import scipy as sp
import numpy as np
import os
import sys
import time
import copy

from _QUICGraphicalLassoCV import QUICGraphicalLassoCV

import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress *all* warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

_numpy2ri_converter = numpy2ri.converter



def make_cov(p, data_type, seed=1):
    p = int(p)
    seed = int(seed)
    elapsed_time = np.nan

    try:
        r = robjects.r
        r('''suppressMessages({library(huge)})''')

        robjects.globalenv["p"] = p
        robjects.globalenv["seed_rng"] = seed

        if data_type == "random":
            timed_result = r("""
                system.time({
                    suppressMessages(suppressWarnings({
                        set.seed(seed_rng)
                        out <- huge.generator(
                            n = 2,
                            d = p,
                            graph = "random",
                            verbose = FALSE
                        )
                    }))
                })
            """)

        elif data_type == "hub":
            timed_result = r("""
                system.time({
                    suppressMessages(suppressWarnings({
                        set.seed(seed_rng)
                        out <- huge.generator(
                            n = 2,
                            d = p,
                            graph = "hub",
                            verbose = FALSE
                        )
                    }))
                })
            """)

        elif data_type == "cluster":
            timed_result = r("""
                system.time({
                    suppressMessages(suppressWarnings({
                        set.seed(seed_rng)
                        out <- huge.generator(
                            n = 2,
                            d = p,
                            graph = "cluster",
                            verbose = FALSE
                        )
                    }))
                })
            """)

        elif data_type == "band":
            timed_result = r("""
                system.time({
                    suppressMessages(suppressWarnings({
                        set.seed(seed_rng)
                        out <- huge.generator(
                            n = 2,
                            d = p,
                            graph = "band",
                            verbose = FALSE
                        )
                    }))
                })
            """)

        elif data_type == "sf":
            timed_result = r("""
                system.time({
                    suppressMessages(suppressWarnings({
                        set.seed(seed_rng)
                        out <- huge.generator(
                            n = 2,
                            d = p,
                            graph = "scale-free",
                            verbose = FALSE
                        )
                    }))
                })
            """)

        else:
            raise ValueError("Invalid data_type. Choose from 'random', 'hub', 'cluster', 'band', or 'sf'.")

        elapsed_time = float(np.asarray(timed_result)[2])

        with robjects.default_converter.context():
            out = robjects.globalenv["out"]
            iC = np.asarray(out.rx2("omega"), dtype=float)

        iC = np.round(iC, 12)
        C = np.linalg.inv(iC)

    except Exception:
        iC = np.eye(p) * np.nan
        C = np.eye(p) * np.nan
        elapsed_time = np.nan

    #print("R runtime: ", elapsed_time)

    return iC, C

def make_samples(cov, mean=None, n=None, df=np.inf, seed=0):
    if mean is None:
        mean = np.zeros(cov.shape[0])
    if n is None:
        n = 100

    rng = np.random.default_rng(seed)  # Create a reproducible generator

    Y = sp.stats.multivariate_t.rvs(
        loc=mean, shape=cov, df=df, size=n, random_state=rng).T

    return Y

def normalize_data(Y):
    sd = Y.std(axis=1, keepdims=True)
    D = np.diag(1.0 / sd.flatten())
    iD = np.diag(sd.flatten())
    Y = D @ (Y - Y.mean(axis=1, keepdims=True))  # Normalized Y

    return Y, D, iD

def compute_scov(Y):

    p, n = Y.shape
    runtime = time.time()

    C = np.cov(Y, bias=False)

    if n > p:
        iC = np.linalg.inv(C)
    else:
        iC = np.linalg.pinv(C)

    runtime = time.time() - runtime

    # Compile results into a dictionary
    result = {
        'iC': copy.deepcopy(iC),
        'C':  copy.deepcopy(C),
        'runtime': runtime
    }
    return SimpleNamespace(**result)

def compute_ledoit(Y):
    runtime = time.time()

    res = LedoitWolf(store_precision=True).fit(Y.T)
    C = res.covariance_
    iC = res.precision_
    rho = res.shrinkage_

    runtime = time.time() - runtime

    # Compile results into a dictionary
    result = {
        'iC': copy.deepcopy(iC),
        'C':  copy.deepcopy(C),
        'l': rho,
        'runtime': runtime
    }
    return SimpleNamespace(**result)

def compute_aquic(Y, c=None, gamma=None, k=None, tol=1e-4, max_iter=100, L_ii=1e-6, verbose=0, clip_gamma=True):

    p, n = Y.shape

    # Set default for c if not provided
    c_max = np.floor(p/2)
    if c is None:
        c = 30
    c = np.clip(c, 1, c_max)

    if gamma is None:
        val   = 1. - c / p
        gamma = 0.5 * (1. - sp.special.erf(2. * sp.special.erfinv(val)))
    else:
        gamma = gamma

    if clip_gamma == True:
        gamma = np.clip(gamma, 1e-16, 0.5 - 1e-16)
    else:
        print(f'Gamma is set as is !! {gamma}')

    if k is None:
        k = n/2
    k = np.clip(k, 1, n)

    runtime = time.time()

    Y = np.array(np.ascontiguousarray(Y, dtype=np.float64), order='F')

    X, W = quic_pybind.AQUIC(Y,
                             float(k),          # C++ double
                             float(gamma),      # C++ double
                             float(tol),        # C++ double
                             int(max_iter),
                             float(L_ii),       # C++ double
                             int(verbose)       # C++ int
                             )

    runtime = time.time() - runtime

    print(f"- gamma: {gamma}, c: {c}, p: {p}, n: {n} n/k: {n/k}, nnzpriC: {np.count_nonzero(X)/p}  ||iC||: {np.linalg.norm(X, ord='fro')} ||C||: {np.linalg.norm(W, ord='fro')} ")

    # Compile results into a dictionary
    result = {
        'iC': X,
        'C': W,
        'X': X,
        'W': W,
        'runtime': runtime
    }
    return SimpleNamespace(**result)

def compute_aquic_q(Y, c=None, gamma=None, k=None, tol=1e-4, max_iter=100, L_ii=1e-6, verbose=0, clip_gamma=True):

    p, n = Y.shape

    # Set default for c if not provided
    c_max = np.floor(p/2)
    if c is None:
        c = 30
    c = np.clip(c, 1, c_max)

    if gamma is None:
        #val = 1. - c / (2. * (p-c-1.))
        val   = 1. - c / p
        gamma = 0.5 * (1. - sp.special.erf(2. * sp.special.erfinv(val)))
    else:
        gamma = gamma

    if clip_gamma == True:
        gamma = np.clip(gamma, 1e-16, 0.5 - 1e-16)
    else:
        print(f'Gamma is set as is !! {gamma}')

    if k is None:
        k = n/2
    k = np.clip(k, 1, n)

    runtime = time.time()

    Y = np.array(np.ascontiguousarray(Y, dtype=np.float64), order='F')

    X, W = quic_pybind.AQUIC_q(Y,
                             float(k),          # C++ double
                             float(gamma),      # C++ double
                             float(tol),        # C++ double
                             int(max_iter),
                             float(L_ii),       # C++ double
                             int(verbose)       # C++ int
                             )

    runtime = time.time() - runtime

    print(f"- gamma: {gamma}, c: {c}, p: {p}, n: {n} n/k: {n/k}, nnzpriC: {np.count_nonzero(X)/p}  ||iC||: {np.linalg.norm(X, ord='fro')} ||C||: {np.linalg.norm(W, ord='fro')} ")

    # Compile results into a dictionary
    result = {
        'iC': X,
        'C': W,
        'X': X,
        'W': W,
        'runtime': runtime
    }
    return SimpleNamespace(**result)

def compute_tiger(Y):

    p, n = Y.shape
    # Set maximum number of iterations and tolerance for convergence

    # Normalize data if specified
    Y, D, iD = normalize_data(Y)

    # Transpose the NumPy array (as you're passing X.T to R)
    Y_T = Y.T

    try:
        # Import R base and 'flare' library
        r = robjects.r
        r('''suppressMessages({library(flare)})''')
       
        with localconverter(robjects.default_converter + numpy2ri.converter):
            # Assign the transposed data to a variable in the R environment
            robjects.globalenv["D_data"] = Y_T

        # Now use `D_data` in the R environment and run `sugm`
        # Using R's system.time() to time the execution of the sugm function
        timed_result = r("""
            system.time({
                invisible(capture.output({
                    suppressMessages(suppressWarnings({
                        out <- sugm(D_data, method = "tiger")
                        sel <- sugm.select(out, criterion = "cv")
                    }))
                }))
            })
        """)

        # Extract the timing result from R
        user_time = timed_result[0]  # User CPU time
        system_time = timed_result[1]  # System CPU time
        elapsed_time = timed_result[2]  # Total elapsed time

        # Retrieve the 'out' object from the R environment and convert to a Python dictionary
        with robjects.default_converter.context():
            sel = robjects.globalenv["sel"]
            iC = np.asarray(sel.rx2("opt.icov"), dtype=float)
            l = np.asarray(sel.rx2("opt.lambda"), dtype=float)[0]

        # De-normalize inverse covariance matrix if normalization was applied
        iC = D @ iC @ D
        C = np.linalg.pinv(iC)

    except:
        iC = np.eye(p) * np.nan
        C = np.eye(p) * np.nan
        l = np.nan
        elapsed_time = np.nan

    # Compile results into a dictionary
    result = {
        'iC': copy.deepcopy(iC),
        'C': copy.deepcopy(C),
        'l': l,
        'runtime': elapsed_time
    }
    return SimpleNamespace(**result)



def compute_clime(Y):

    p, n = Y.shape
    # Set maximum number of iterations and tolerance for convergence

    # Normalize data if specified
    Y, D, iD = normalize_data(Y)

    # Transpose the NumPy array (as you're passing X.T to R)
    Y_T = Y.T

    try:
        # Import R base and 'flare' library
        r = robjects.r
        r('''suppressMessages({library(flare)})''')
       
        with localconverter(robjects.default_converter + numpy2ri.converter):
            # Assign the transposed data to a variable in the R environment
            robjects.globalenv["D_data"] = Y_T

        # Now use `D_data` in the R environment and run `sugm`
        # Using R's system.time() to time the execution of the sugm function
        timed_result = r("""
            system.time({
                invisible(capture.output({
                    suppressMessages(suppressWarnings({
                        out <- sugm(D_data, method = "clime")
                        sel <- sugm.select(out, criterion = "cv")
                    }))
                }))
            })
        """)

        # Extract the timing result from R
        user_time = timed_result[0]  # User CPU time
        system_time = timed_result[1]  # System CPU time
        elapsed_time = timed_result[2]  # Total elapsed time

        # Retrieve the 'out' object from the R environment and convert to a Python dictionary
        with robjects.default_converter.context():
            sel = robjects.globalenv["sel"]
            iC = np.asarray(sel.rx2("opt.icov"), dtype=float)
            l = np.asarray(sel.rx2("opt.lambda"), dtype=float)[0]

        # De-normalize inverse covariance matrix if normalization was applied
        iC = D @ iC @ D
        C = np.linalg.pinv(iC)

    except:
        iC = np.eye(p) * np.nan
        C = np.eye(p) * np.nan
        l = np.nan
        elapsed_time = np.nan

    # Compile results into a dictionary
    result = {
        'iC': copy.deepcopy(iC),
        'C': copy.deepcopy(C),
        'l': l,
        'runtime': elapsed_time
    }
    return SimpleNamespace(**result)


def compute_glasso_cv(Y):

    p, n = Y.shape

    # Normalize data if specified
    Y, D, iD = normalize_data(Y)

    try:
        runtime = time.time()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            warnings.simplefilter("ignore", category=RuntimeWarning)

            # Initialize GraphicalLassoCV
            model = GraphicalLassoCV(n_jobs=8, tol=1e-4, max_iter=100,eps=1e-4)
            model.fit(Y.T)  # Fit the model on transposed data

        iC = model.precision_
        C = model.covariance_

        # De-normalize inverse covariance matrix if normalization was applied
        iC = D @ iC @ D
        C = iD @ C @ iD
        l = model.alpha_
        runtime = time.time() - runtime
    except:
        iC = np.eye(p) * np.nan
        C = np.eye(p) * np.nan
        l = np.nan
        runtime = np.nan

    # Compile results into a dictionary
    result = {
        'iC': copy.deepcopy(iC),
        'C':  copy.deepcopy(C),
        'l': l,
        'runtime': runtime
    }
    return SimpleNamespace(**result)

def compute_quic_cv(Y):

    p, n = Y.shape

    # Normalize data if specified
    Y, D, iD = normalize_data(Y)

    try:
        runtime = time.time()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            warnings.simplefilter("ignore", category=RuntimeWarning)

            # Initialize QUICGraphicalLassoCV
            model = QUICGraphicalLassoCV(n_jobs=8, tol=1e-4, max_iter=100)
            model.fit(Y.T)  # Fit the model on transposed data

        iC = model.precision_
        C = model.covariance_

        # De-normalize inverse covariance matrix if normalization was applied
        iC = D @ iC @ D
        C = iD @ C @ iD
        l = model.alpha_
        runtime = time.time() - runtime
    except:
        iC = np.eye(p) * np.nan
        C = np.eye(p) * np.nan
        l = np.nan
        runtime = np.nan

    # Compile results into a dictionary
    result = {
        'iC': copy.deepcopy(iC),
        'C':  copy.deepcopy(C),
        'l': l,
        'runtime': runtime
    }
    return SimpleNamespace(**result)

def compute_glasso_rho(Y, rho):
    p, n = Y.shape

    # Normalize data if specified
    Y, D, iD = normalize_data(Y)

    start_time = time.time()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)

        # Initialize GraphicalLasso with the provided tuning parameter 'rho'
        model = GraphicalLasso(alpha=rho, tol=1e-4, max_iter=100)
        model.fit(Y.T)  # Fit the model on the transposed data

    iC = model.precision_
    C = model.covariance_
    runtime = time.time() - start_time

    # De-normalize inverse covariance matrix if normalization was applied
    iC = D @ iC @ D
    C = iD @ C @ iD

    # Compile results into a dictionary
    result = {
        'iC': copy.deepcopy(iC),
        'C': copy.deepcopy(C),
        'l': rho,
        'runtime': runtime
    }
    return SimpleNamespace(**result)

def compute_metrics(result, C_true, iC_true, Y):

    if result is None or np.isnan(np.sum(result.iC)):
        result_metrics = {
            'err_C': np.nan,
            'err_iC': np.nan,
            'err_S': np.nan,
            'nnz': np.nan,
            'nll': np.nan,
            'sphr': np.nan,
            'precision': np.nan,
            'recall': np.nan,
            'f1': np.nan,
            'kl': np.nan,
            'aic':np.nan,
            'runtime': np.nan
        }
        return SimpleNamespace(**result_metrics)

    p, n = Y.shape
    S   = np.cov(Y, bias=True)
    C   = result.C #+ np.eye(p) * 1e-6
    iC  = result.iC #+ np.eye(p) * 1e-6
    
    if C_true is None or np.all(np.isnan(C_true)):
        C_true = np.ones((p, p)) * np.nan

    if iC_true is None or np.all(np.isnan(iC_true)):
        iC_true = np.ones((p, p)) * np.nan

    # Compute error metrics on covariance/precision matrices
    err_C  = np.linalg.norm(C  - C_true, ord="fro") / np.linalg.norm(C_true, ord="fro")
    err_iC = np.linalg.norm(iC - iC_true, ord="fro") / np.linalg.norm(iC_true, ord="fro")
    err_S  = np.linalg.norm(C  - S, ord="fro")
    nnz    = (np.count_nonzero(iC)) / p
    
    log_det_sign, logdet_iC = np.linalg.slogdet(iC)
    if log_det_sign <= 0:
        print("Warning: iC is not SPD.")
    nll = -1.0 * logdet_iC + np.sum(iC * S)  # sum(iC * S) = trace(iC@C)

    sphr   = p * np.sum(iC * iC) / np.trace(iC)**2 - 1  # sum(iC * iC) = trace(iC@iC)
    aic    =  nll + nnz/2 + p

    # Compute KL divergence loss for two zero-mean Gaussians.
    tr_term = float(np.sum(C_true * iC)) # = trace(C_true @ iC )
    log_det_sign, logdet_prod = np.linalg.slogdet(C_true @ iC)  # should be SPD ⇒ s>0
    if log_det_sign <= 0:
        print("Error C_true @ iC is not SPD.")
    kl = tr_term - logdet_prod - p

    # Compute F1 score for structure recovery of the precision matrix.
    # Threshold the absolute values to define nonzero connections.
    mask = np.triu(np.ones((p, p), dtype=bool), k=1)

    # Binarize supports on masked entries
    y_true = (np.abs(iC_true) > 0)[mask]
    y_pred = (np.abs(iC) > 0.0)[mask]

    # Counts
    tp = int(np.count_nonzero(y_pred &  y_true))
    fp = int(np.count_nonzero(y_pred & ~y_true))
    fn = int(np.count_nonzero(~y_pred &  y_true))
    tn = int(np.count_nonzero(~y_pred & ~y_true))

    # Metrics (zero-safe)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    
    # Compile all computed metrics into a structured object.
    result_metrics = {
        'err_C': err_C,
        'err_iC': err_iC,
        'err_S': err_S,
        'nnz': nnz,
        'nll': nll,
        'sphr': sphr,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'kl': kl,
        'aic':aic,
        'runtime': result.runtime
    }
    return SimpleNamespace(**result_metrics)