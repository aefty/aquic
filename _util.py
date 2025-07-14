import quic_pybind
from rpy2.robjects import numpy2ri
import rpy2.robjects as robjects
from gglasso.problem import glasso_problem
from types import SimpleNamespace
from sklearn.covariance import LedoitWolf
from sklearn.covariance import GraphicalLassoCV, GraphicalLasso
import networkx as nx
import pandas as pd
import scipy as sp
import numpy as np
import os
import sys
import time
import copy

import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress *all* warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


numpy2ri.activate()


def make_cov(p, data_type, seed=1):

    np.random.seed(seed)

    r = robjects.r
    r.assign("p", p)
    r.assign("seed_rng", seed)

    if data_type == "random":

        timed_result = r('''
            library(huge)
            set.seed(seed_rng);
            system.time({
            suppressMessages(suppressWarnings({
                out <- huge.generator(n = 2, d = p,
                                      graph = "random", verbose = FALSE)
            }))
            })
        ''')
    elif data_type == "hub":

        timed_result = r('''
            library(huge)
            set.seed(seed_rng)
            system.time({
            suppressMessages(suppressWarnings({
                out <- huge.generator(n = 2, d = p,
                                      graph = "hub", verbose = FALSE)
            }))
            })
        ''')

    elif data_type == "cluster":

        timed_result = r('''
            library(huge)
            set.seed(seed_rng)
            system.time({
            suppressMessages(suppressWarnings({
                out <- huge.generator(n = 2, d = p,
                                      graph = "cluster", verbose = FALSE)
            }))
            })
        ''')
    elif data_type == "band":

        timed_result = r('''
            library(huge)
            set.seed(seed_rng)
            system.time({
            suppressMessages(suppressWarnings({
                out <- huge.generator(n = 2, d = p,
                                      graph = "band", verbose = FALSE)
            }))
            })
        ''')
    elif data_type == "sf":

        timed_result = r('''
            library(huge)
            set.seed(seed_rng)
            system.time({
            suppressMessages(suppressWarnings({
                out <- huge.generator(n = 2, d = p,
                                      graph = "scale-free", verbose = FALSE)
            }))
            })
        ''')
    else:
        raise ValueError("Invalid mode. Choose from 'diag', or 'pca'.")

    temp = r['out'].rx2('omega')

    iC = np.round(np.array(temp), 12)
    C = np.linalg.inv(iC)

    return iC, C


def make_samples(cov, mean=None, n=None, df=np.inf, seed=0):
    if mean is None:
        mean = np.zeros(cov.shape[0])
    if n is None:
        n = 100
    Y = sp.stats.multivariate_t.rvs(
        loc=mean, shape=cov, df=df, size=n, random_state=seed).T

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


def compute_aquic(Y, c=None, gamma=None, k=None, tol=1e-3, max_iter=100, verbose=0):

    p, n = Y.shape

    # Set default for c if not provided
    c_max = (2/3) * (p-1)
    if c is None:
        c = 40.0
    c = np.clip(c, 1, c_max-1)

    gamma = 0.5*(1. - sp.special.erf(2. *
                 sp.special.erfinv(1. - c / (2. * (p-c-1.)))))
    gamma = np.clip(gamma, 1e-10, 0.5 - 1e-6)

    if k is None:
        k = n/2
    k = np.clip(k, 1, n)

    print(f"- gamma: {gamma}, c: {c}, k: {k}, p: {p}, n: {n}")

    runtime = time.time()

    Y = np.array(np.ascontiguousarray(Y, dtype=np.float64), order='F')

    X, W = quic_pybind.AQUIC(Y, k, gamma, tol, max_iter, verbose)
    runtime = time.time() - runtime

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
        elapsed_time = 0

        # Import R base and 'flare' library
        r = robjects.r

        # Ensure the 'flare' package is installed and loaded in the R environment
        r('''suppressMessages({library(flare)})''')

        # Assign the transposed data to a variable in the R environment
        r.assign("D_data", Y_T)

        # Now use `D_data` in the R environment and run `sugm`
        # Using R's system.time() to time the execution of the sugm function
        timed_result = r('''
            system.time({
            invisible(capture.output({
            suppressMessages(suppressWarnings({
                out <- sugm(D_data, method = "tiger", prec = 1e-4, max.ite = 100)
                sel <- sugm.select(out, criterion = "cv")
            }))
            }))
            })
        ''')

        # Extract the timing result from R
        user_time = timed_result[0]  # User CPU time
        system_time = timed_result[1]  # System CPU time
        elapsed_time = timed_result[2]  # Total elapsed time

        # Retrieve the 'out2' object from the R environment and convert to a Python dictionary
        out = r['sel']

        iC = np.array(np.array(out.rx2('opt.icov')))
        l = np.array(np.array(out.rx2('opt.lambda')))

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
        elapsed_time = 0
        # Import R base and 'flare' library
        r = robjects.r

        # Ensure the 'flare' package is installed and loaded in the R environment
        r('''suppressMessages({library(flare)})''')

        # Assign the transposed data to a variable in the R environment
        r.assign("D_data", Y_T)

        # Now use `D_data` in the R environment and run `sugm`
        # Using R's system.time() to time the execution of the sugm function
        timed_result = r('''
            system.time({
                invisible(capture.output({
                suppressMessages(suppressWarnings({
                out <- sugm(D_data, method = "clime",prec = 1e-4, max.ite = 100);
                sel <- sugm.select(out, criterion = "cv");
            }))
            }))
            })
        ''')

        # Extract the timing result from R
        user_time = timed_result[0]  # User CPU time
        system_time = timed_result[1]  # System CPU time
        elapsed_time = timed_result[2]  # Total elapsed time

        # Retrieve the 'out2' object from the R environment and convert to a Python dictionary
        out = r['sel']

        iC = np.array(np.array(out.rx2('opt.icov')))
        l = np.array(np.array(out.rx2('opt.lambda')))

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


def compute_glasso(Y):

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
            model = GraphicalLassoCV(n_jobs=-1, tol=1e-4, max_iter=100)
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
