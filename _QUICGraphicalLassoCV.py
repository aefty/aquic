"""
QUICGraphicalLassoCV
--------------------
Cross-validated precision-matrix estimator using the QUIC solver.

API mirrors sklearn.covariance.GraphicalLassoCV:
  - same constructor parameters (alphas, n_refinements, cv, tol, max_iter,
    n_jobs, verbose, assume_centered, eps)
  - same fitted attributes (covariance_, precision_, alpha_, cv_results_,
    location_)
  - input convention: fit(X) with X shaped (n_samples, n_features)

Extra parameter vs GraphicalLassoCV:
  - diag_penalty : float, default 0.0
      Diagonal penalty passed to QUIC as L_ii.  Set to 0.0 for strict
      equivalence with the sklearn penalty.
"""

from __future__ import annotations

import operator
import time
import warnings
from typing import Optional, Sequence, Tuple, Union

import numpy as np
from joblib import Parallel, delayed
from sklearn.covariance import EmpiricalCovariance, empirical_covariance, log_likelihood
from sklearn.model_selection import KFold, check_cv, cross_val_score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _alpha_max(emp_cov: np.ndarray) -> float:
    """Largest alpha for which the precision matrix is diagonal.

    Mirrors sklearn's alpha_max: max of |off-diagonal empirical cov|.
    """
    A = np.abs(emp_cov)
    np.fill_diagonal(A, 0.0)
    return float(A.max())


def _center(X: np.ndarray, assume_centered: bool) -> np.ndarray:
    if assume_centered:
        return X
    return X - X.mean(axis=0, keepdims=True)


def _quic_solve(
    X_sn: np.ndarray,
    alpha: float,
    *,
    tol: float,
    max_iter: int,
    diag_penalty: float,
    verbose: int,
    assume_centered: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run one QUIC solve.  Returns (covariance, precision), both (p, p)."""
    import quic_pybind

    Xc = _center(np.asarray(X_sn, dtype=np.float64), assume_centered)
    # Scale by sqrt(n/(n-1)) so QUIC's internal S = XcXc^T/n matches
    # sklearn's empirical_covariance which divides by (n-1).
    n = Xc.shape[0]
    if n > 1:
        Xc = Xc * np.sqrt(n / (n - 1))
    # QUIC binding expects Y shaped (p, n) = (n_features, n_samples)
    Y = np.array(np.ascontiguousarray(Xc.T, dtype=np.float64), order='F')

    iC, C = quic_pybind.QUIC(
        Y,
        float(alpha),
        float(tol),
        int(max_iter),
        float(diag_penalty),
        int(verbose),
    )

    return np.asarray(C, dtype=float), np.asarray(iC, dtype=float)

def _quic_path(
    X_train: np.ndarray,
    alphas: np.ndarray,
    X_test: Optional[np.ndarray],
    *,
    tol: float,
    max_iter: int,
    diag_penalty: float,
    verbose: int,
    assume_centered: bool,
):
    """Solve QUIC for a sequence of alphas on one CV fold.

    Returns
    -------
    covariances : list of (p, p) arrays
    precisions  : list of (p, p) arrays
    scores      : list of float  (only when X_test is not None)
    """
    test_emp_cov = None
    if X_test is not None:
        test_emp_cov = empirical_covariance(X_test, assume_centered=assume_centered)

    covariances, precisions, scores = [], [], []

    for alpha in alphas:
        cov, prec = _quic_solve(
            X_train, float(alpha),
            tol=tol, max_iter=max_iter,
            diag_penalty=diag_penalty,
            verbose=verbose,
            assume_centered=assume_centered,
        )
        covariances.append(cov)
        precisions.append(prec)

        if test_emp_cov is not None:
            s = log_likelihood(test_emp_cov, prec)
            scores.append(float(s) if np.isfinite(s) else -np.inf)

    if X_test is not None:
        return covariances, precisions, scores
    return covariances, precisions


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class QUICGraphicalLassoCV:
    """Cross-validated graphical lasso using the QUIC solver.

    Parameters
    ----------
    alphas : int or sequence of float, default 4
        If int: number of alpha values on the initial log-spaced grid.
        If sequence: explicit alpha values to try (n_refinements is ignored).
    n_refinements : int, default 4
        Number of grid-refinement rounds (same logic as GraphicalLassoCV).
    cv : int, cross-validation generator, or iterable, default None
        Determines the cross-validation split strategy.
        None → 5-fold.
    tol : float, default 1e-4
        Convergence tolerance for QUIC.
    max_iter : int, default 100
        Maximum QUIC iterations for the final refit.
        CV folds use max(1, max_iter // 10) (same budget as sklearn).
    n_jobs : int or None, default None
        Number of jobs for joblib parallelism over CV folds.
    verbose : int, default 0
        Verbosity level.
    assume_centered : bool, default False
        If True, data are not mean-subtracted before fitting.
    eps : float, default machine epsilon
        Lower-bound on alpha to avoid degenerate solutions.
    diag_penalty : float, default 0.0
        QUIC diagonal penalty (L_ii).  0.0 → strict sklearn parity.

    Attributes
    ----------
    covariance_ : ndarray, shape (p, p)
    precision_  : ndarray, shape (p, p)
    alpha_      : float  — selected regularisation strength
    cv_results_ : dict   — keys: 'alphas', 'mean_test_score', 'std_test_score',
                           'split{k}_test_score'
    location_   : ndarray, shape (p,) — training-set mean (or zeros)
    """

    def __init__(
        self,
        *,
        alphas: Union[int, Sequence[float]] = 4,
        n_refinements: int = 4,
        cv=None,
        tol: float = 1e-4,
        max_iter: int = 100,
        n_jobs=None,
        verbose: int = 0,
        assume_centered: bool = False,
        eps: float = 1e-4, #np.finfo(np.float64).eps,
        diag_penalty: float =0.0,
    ):
        self.alphas = alphas
        self.n_refinements = n_refinements
        self.cv = cv
        self.tol = tol
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.assume_centered = assume_centered
        self.eps = eps
        self.diag_penalty = diag_penalty

    # ------------------------------------------------------------------
    def fit(self, X, y=None):
        """Fit the model by cross-validating the regularisation parameter.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : ignored
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D; got shape {X.shape}.")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or Inf.")

        n_samples, n_features = X.shape

        self.location_ = np.zeros(n_features) if self.assume_centered else X.mean(axis=0)

        emp_cov = empirical_covariance(X, assume_centered=self.assume_centered)
        cv = check_cv(self.cv, y, classifier=False)
        inner_verbose = max(0, int(self.verbose) - 1)
        cv_max_iter = max(1, self.max_iter // 10)

        # ---- build initial alpha grid --------------------------------
        alphas_is_grid = hasattr(self.alphas, '__len__')
        if alphas_is_grid:
            alphas = np.asarray(self.alphas, dtype=float)
            n_refinements = 1
        else:
            n_grid = int(self.alphas)
            n_refinements = int(self.n_refinements)
            a_max = _alpha_max(emp_cov)
            a_min = max(self.eps, 1e-2 * a_max)
            alphas = np.logspace(np.log10(a_min), np.log10(a_max), n_grid)[::-1]

        path: list = []          # accumulated (alpha, scores_tuple, covs_tuple)
        t0 = time.time()

        for refine_i in range(n_refinements):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                fold_results = Parallel(n_jobs=self.n_jobs, verbose=int(self.verbose))(
                    delayed(_quic_path)(
                        X[train], alphas, X[test],
                        tol=float(self.tol),
                        max_iter=cv_max_iter,
                        diag_penalty=float(self.diag_penalty),
                        verbose=self.verbose,
                        assume_centered=self.assume_centered,
                    )
                    for train, test in cv.split(X, y)
                )

            # fold_results : list over folds of (covs, precs, scores)
            fold_covs   = [fr[0] for fr in fold_results]   # (n_folds, n_alphas, p, p)
            fold_scores = [fr[2] for fr in fold_results]   # (n_folds, n_alphas)

            # transpose to (n_alphas, n_folds)
            scores_by_alpha = list(zip(*fold_scores))
            covs_by_alpha   = list(zip(*fold_covs))

            path.extend(zip(alphas, scores_by_alpha, covs_by_alpha))
            path.sort(key=operator.itemgetter(0), reverse=True)   # descending alpha

            # ---- find best alpha -------------------------------------
            best_score = -np.inf
            best_index = 0
            last_finite_idx = 0

            for idx, (a, sc, _covs) in enumerate(path):
                mean_sc = float(np.mean(sc))
                if mean_sc >= 0.1 / np.finfo(np.float64).eps:
                    mean_sc = np.nan
                if np.isfinite(mean_sc):
                    last_finite_idx = idx
                if np.isfinite(mean_sc) and mean_sc >= best_score:
                    best_score = mean_sc
                    best_index = idx

            # ---- refine grid around best_index ----------------------
            if not alphas_is_grid:
                if best_index == 0:
                    a_hi, a_lo = path[0][0], path[1][0]
                elif best_index == last_finite_idx and best_index != len(path) - 1:
                    a_hi, a_lo = path[best_index][0], path[best_index + 1][0]
                elif best_index == len(path) - 1:
                    a_hi = path[best_index][0]
                    a_lo = max(self.eps, 0.01 * a_hi)
                else:
                    a_hi, a_lo = path[best_index - 1][0], path[best_index + 1][0]

                alphas = np.logspace(np.log10(a_hi), np.log10(a_lo), n_grid + 2)[1:-1]

            if self.verbose and n_refinements > 1:
                print(
                    f"[QUICGraphicalLassoCV] refinement {refine_i + 1}/{n_refinements} "
                    f"— {time.time() - t0:.1f}s  best_alpha={path[best_index][0]:.4e}"
                )

        # ---- build cv_results_ (sklearn-compatible) -----------------
        path_alphas  = [a  for a, _, __ in path]
        path_scores  = [sc for _, sc, __ in path]    # each is a tuple/array over folds

        # append alpha=0 baseline (EmpiricalCovariance), same as sklearn
        path_alphas.append(0.0)
        path_scores.append(
            cross_val_score(
                EmpiricalCovariance(assume_centered=self.assume_centered),
                X, cv=cv, n_jobs=self.n_jobs, verbose=inner_verbose,
            )
        )

        grid_scores = np.asarray([np.asarray(s) for s in path_scores])  # (n_alphas+1, n_folds)

        self.cv_results_ = {"alphas": np.asarray(path_alphas)}
        for k in range(grid_scores.shape[1]):
            self.cv_results_[f"split{k}_test_score"] = grid_scores[:, k]
        self.cv_results_["mean_test_score"] = grid_scores.mean(axis=1)
        self.cv_results_["std_test_score"]  = grid_scores.std(axis=1)

        self.alpha_ = float(path_alphas[best_index])

        # ---- final refit on all data --------------------------------
        if self.alpha_ == 0.0:
            ec = EmpiricalCovariance(assume_centered=self.assume_centered).fit(X)
            self.covariance_ = ec.covariance_
            self.precision_  = ec.precision_
            return self

        cov, prec = _quic_solve(
            X, self.alpha_,
            tol=float(self.tol),
            max_iter=int(self.max_iter),
            diag_penalty=float(self.diag_penalty),
            verbose=inner_verbose,
            assume_centered=self.assume_centered,
        )
        self.covariance_ = cov
        self.precision_  = prec
        return self

    # ------------------------------------------------------------------
    def score(self, X_test, y=None):
        """Log-likelihood of X_test under the fitted model."""
        test_emp_cov = empirical_covariance(
            np.asarray(X_test, dtype=float), assume_centered=self.assume_centered
        )
        return log_likelihood(test_emp_cov, self.precision_)
