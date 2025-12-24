from __future__ import annotations

import operator
import time
import warnings
from numbers import Real
from typing import Optional, Tuple

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import _fit_context
from sklearn.covariance import EmpiricalCovariance, empirical_covariance, log_likelihood
from sklearn.covariance._graph_lasso import BaseGraphicalLasso, alpha_max  # sklearn 1.4.2
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import check_cv, cross_val_score
from sklearn.utils.validation import _is_arraylike_not_scalar, check_scalar


def _center_like_sklearn(X_sn: np.ndarray, assume_centered: bool) -> np.ndarray:
    """Match sklearn empirical_covariance centering semantics.

    Parameters
    ----------
    X_sn : array, shape (n_samples, n_features)
    """
    X_sn = np.asarray(X_sn, dtype=float)
    if X_sn.ndim != 2:
        raise ValueError(f"X must be 2D; got shape {X_sn.shape}.")
    if assume_centered:
        return X_sn
    return X_sn - X_sn.mean(axis=0, keepdims=True)


def _quic_solve_from_data(
    X_sn: np.ndarray,
    alpha: float,
    *,
    tol: float,
    max_iter: int,
    verbose: int,
    diag_penalty: float,
    assume_centered: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve Graphical Lasso using QUIC.

    Public input convention (aligned with sklearn):
      - X_sn is (n_samples, n_features) = (n, p).

    QUIC binding input convention (hard-coded here):
      - Y is (n_features, n_samples) = (p, n).

    Calls:
        iC, C = quic_pybind.QUIC(Y, L_ij, tol, max_iter, L_ii, verbose)

    Returns
    -------
    cov : (p, p)
    prec: (p, p)
    """
    Xc_sn = _center_like_sklearn(X_sn, assume_centered=assume_centered)
    n_samples, n_features = Xc_sn.shape

    # QUIC input Y must be (p, n) = (n_features, n_samples)
    Y = np.asfortranarray(Xc_sn.T, dtype=np.float64)

    import quic_pybind  # hard-coded per your request

    iC, C = quic_pybind.QUIC(
        Y,
        float(alpha),
        float(tol),
        int(max_iter),
        float(diag_penalty),
        int(verbose),
    )

    prec = np.asarray(iC, dtype=float)
    cov = np.asarray(C, dtype=float)

    if prec.shape != (n_features, n_features):
        raise ValueError(
            f"QUIC returned precision with shape {prec.shape}; expected ({n_features}, {n_features}). "
            "This indicates the binding is not interpreting Y as (n_features, n_samples)."
        )
    if cov.shape != (n_features, n_features):
        raise ValueError(
            f"QUIC returned covariance with shape {cov.shape}; expected ({n_features}, {n_features})."
        )

    return cov, prec


def _quic_path(
    X_sn: np.ndarray,
    *,
    alphas: np.ndarray,
    X_test_sn: Optional[np.ndarray],
    tol: float,
    enet_tol: float,  # kept for signature parity
    max_iter: int,
    verbose: int,
    diag_penalty: float,
    assume_centered: bool,
):
    """Path analogue of sklearn.graphical_lasso_path but solved via QUIC."""
    _ = enet_tol  # API parity

    covariances_: list[np.ndarray] = []
    precisions_: list[np.ndarray] = []
    scores_: list[float] = []

    test_emp_cov = None
    if X_test_sn is not None:
        test_emp_cov = empirical_covariance(X_test_sn, assume_centered=assume_centered)

    for alpha in alphas:
        cov, prec = _quic_solve_from_data(
            X_sn,
            float(alpha),
            tol=tol,
            max_iter=max_iter,
            verbose=max(0, int(verbose) - 1),
            diag_penalty=diag_penalty,
            assume_centered=assume_centered,
        )

        covariances_.append(cov)
        precisions_.append(prec)

        if test_emp_cov is not None:
            s = log_likelihood(test_emp_cov, prec)
            scores_.append(float(s) if np.isfinite(s) else -np.inf)

        if verbose == 1:
            import sys

            sys.stderr.write(".")
            sys.stderr.flush()

    if X_test_sn is not None:
        return covariances_, precisions_, scores_
    return covariances_, precisions_


class QUICGraphicalLassoCV(BaseGraphicalLasso):
    """GraphicalLassoCV-equivalent CV loop (sklearn 1.4.2 logic), but solver is QUIC.

    Input convention: .fit(X) expects X shaped (n, p) = (n_samples, n_features), same as sklearn.

    If your native data storage is X_pn shaped (p, n), call .fit(X_pn.T) for BOTH sklearn
    GraphicalLassoCV and this class.

    For strict equivalence to sklearn GraphicalLassoCV penalty semantics:
      - set diag_penalty = 0.0 (default).
    """

    def __init__(
        self,
        *,
        alphas=4,
        n_refinements=4,
        cv=None,
        tol=1e-4,
        enet_tol=1e-4,
        max_iter=100,
        mode="cd",  # kept for API parity
        n_jobs=None,
        verbose=False,
        eps=np.finfo(np.float64).eps,
        assume_centered=False,
        diag_penalty: float = 0.0,  # L_ii
    ):
        super().__init__(
            tol=tol,
            enet_tol=enet_tol,
            max_iter=max_iter,
            mode=mode,
            verbose=verbose,
            eps=eps,
            assume_centered=assume_centered,
        )
        self.alphas = alphas
        self.n_refinements = n_refinements
        self.cv = cv
        self.n_jobs = n_jobs
        self.diag_penalty = diag_penalty

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        # Public input is (n, p), exactly like sklearn.
        X_sn = np.asarray(X, dtype=float)
        if X_sn.ndim != 2:
            raise ValueError(f"X must be 2D (n, p); got shape {X_sn.shape}.")
        if not np.isfinite(X_sn).all():
            raise ValueError("Input X contains NaN or inf; centering/covariance will be invalid.")

        X_sn = self._validate_data(X_sn, ensure_min_features=2)

        # Same location_ semantics as sklearn (location_ is length p)
        if self.assume_centered:
            self.location_ = np.zeros(X_sn.shape[1])
        else:
            self.location_ = X_sn.mean(axis=0)

        emp_cov = empirical_covariance(X_sn, assume_centered=self.assume_centered)
        cv = check_cv(self.cv, y, classifier=False)

        n_alphas = self.alphas

        if isinstance(self.verbose, bool):
            inner_verbose = int(self.verbose)
        elif isinstance(self.verbose, (int, np.integer)):
            inner_verbose = max(0, int(self.verbose) - 1)
        else:
            inner_verbose = int(bool(self.verbose))

        if _is_arraylike_not_scalar(n_alphas):
            for a in self.alphas:
                check_scalar(
                    a,
                    "alpha",
                    Real,
                    min_val=0,
                    max_val=np.inf,
                    include_boundaries="right",
                )
            alphas = np.asarray(self.alphas, dtype=float)
            n_refinements = 1
        else:
            n_refinements = int(self.n_refinements)
            a1 = alpha_max(emp_cov)
            a0 = 1e-2 * a1
            alphas = np.logspace(np.log10(a0), np.log10(a1), int(n_alphas))[::-1]

        path = []
        t0 = time.time()

        for i in range(n_refinements):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)

                this_path = Parallel(n_jobs=self.n_jobs, verbose=int(self.verbose))(
                    delayed(_quic_path)(
                        X_sn[train],
                        alphas=alphas,
                        X_test_sn=X_sn[test],
                        tol=float(self.tol),
                        enet_tol=float(self.enet_tol),
                        max_iter=max(1, int(0.1 * self.max_iter)),  # exact sklearn CV budget
                        verbose=inner_verbose,
                        diag_penalty=float(self.diag_penalty),
                        assume_centered=self.assume_centered,
                    )
                    for train, test in cv.split(X_sn, y)
                )

            covs, _, scores = zip(*this_path)
            covs = zip(*covs)
            scores = zip(*scores)
            path.extend(zip(alphas, scores, covs))
            path = sorted(path, key=operator.itemgetter(0), reverse=True)

            best_score = -np.inf
            last_finite_idx = 0
            best_index = 0

            for index, (_a, sc, _covs) in enumerate(path):
                this_score = np.mean(sc)
                if this_score >= 0.1 / np.finfo(np.float64).eps:
                    this_score = np.nan
                if np.isfinite(this_score):
                    last_finite_idx = index
                if this_score >= best_score:
                    best_score = this_score
                    best_index = index

            # refine grid (exact sklearn branching)
            if best_index == 0:
                a1 = path[0][0]
                a0 = path[1][0]
            elif best_index == last_finite_idx and best_index != len(path) - 1:
                a1 = path[best_index][0]
                a0 = path[best_index + 1][0]
            elif best_index == len(path) - 1:
                a1 = path[best_index][0]
                a0 = 0.01 * path[best_index][0]
            else:
                a1 = path[best_index - 1][0]
                a0 = path[best_index + 1][0]

            if not _is_arraylike_not_scalar(n_alphas):
                alphas = np.logspace(np.log10(a1), np.log10(a0), int(n_alphas) + 2)[1:-1]

            if self.verbose and n_refinements > 1:
                print(
                    "[QUICGraphicalLassoCV] Done refinement % 2i / %i: % 3is"
                    % (i + 1, n_refinements, time.time() - t0)
                )

        # build cv_results_ (sklearn-compatible structure)
        path = list(zip(*path))
        grid_scores = list(path[1])
        alphas_list = list(path[0])

        # add alpha=0 baseline (sklearn behavior)
        alphas_list.append(0.0)
        grid_scores.append(
            cross_val_score(
                EmpiricalCovariance(assume_centered=self.assume_centered),
                X_sn,
                cv=cv,
                n_jobs=self.n_jobs,
                verbose=inner_verbose,
            )
        )
        grid_scores = np.asarray(grid_scores)

        self.cv_results_ = {"alphas": np.asarray(alphas_list)}
        for k in range(grid_scores.shape[1]):
            self.cv_results_[f"split{k}_test_score"] = grid_scores[:, k]
        self.cv_results_["mean_test_score"] = grid_scores.mean(axis=1)
        self.cv_results_["std_test_score"] = grid_scores.std(axis=1)

        self.alpha_ = float(alphas_list[best_index])

        # final refit on full data using selected alpha
        if self.alpha_ == 0.0:
            ec = EmpiricalCovariance(assume_centered=self.assume_centered).fit(X_sn)
            self.covariance_ = ec.covariance_
            self.precision_ = ec.precision_
            self.n_iter_ = 0
            self.costs_ = None
            return self

        cov, prec = _quic_solve_from_data(
            X_sn,
            self.alpha_,
            tol=float(self.tol),
            max_iter=int(self.max_iter),
            verbose=int(inner_verbose),
            diag_penalty=float(self.diag_penalty),
            assume_centered=self.assume_centered,
        )

        self.covariance_ = cov
        self.precision_ = prec
        self.n_iter_ = None
        self.costs_ = None
        return self


if __name__ == "__main__":
    from sklearn.covariance import GraphicalLassoCV
    from sklearn.model_selection import KFold

    # Reproducibility
    rng = np.random.default_rng(123)

    # Problem size
    p = 25
    n = 300

    # --- generate a sparse SPD precision matrix (Theta_true) ---
    mask = rng.random((p, p)) < 0.08
    W = rng.uniform(-0.25, 0.25, size=(p, p)) * mask
    W = np.triu(W, 1)
    W = W + W.T

    Theta_true = W.copy()
    diag = np.sum(np.abs(Theta_true), axis=1) + 0.5
    np.fill_diagonal(Theta_true, diag)

    Sigma_true = np.linalg.inv(Theta_true)
    L = np.linalg.cholesky(Sigma_true)

    # Native storage as (p, n)
    X = (rng.standard_normal((n, p)) @ L.T).astype(np.float64).T  # (p, n)

   
    cv = KFold(n_splits=5, shuffle=True, random_state=123)

    tol = 1e-4
    max_iter = 100
    alphas = 4
    n_refinements = 4

    gl = GraphicalLassoCV(
        alphas=alphas,
        n_refinements=n_refinements,
        cv=cv,
        tol=tol,
        max_iter=max_iter,
        n_jobs=-1,
        verbose=False,
        assume_centered=False,
    )
    gl.fit(X.T)

    qgl = QUICGraphicalLassoCV(
        alphas=alphas,
        n_refinements=n_refinements,
        cv=cv,
        tol=tol,
        max_iter=max_iter,
        n_jobs=-1,
        verbose=False,
        assume_centered=False,
        diag_penalty=0.0,
    )
    qgl.fit(X.T)

    alpha_sklearn = float(gl.alpha_)
    alpha_quic = float(qgl.alpha_)

    P_sklearn = np.asarray(gl.precision_, dtype=float)
    P_quic = np.asarray(qgl.precision_, dtype=float)

    diff_fro = np.linalg.norm(P_quic - P_sklearn, ord="fro")
    rel_fro = diff_fro / (np.linalg.norm(P_sklearn, ord="fro") + 1e-15)

    print("=== Alpha selection ===")
    print(f"sklearn alpha_: {alpha_sklearn:.8e}")
    print(f"QUIC   alpha_: {alpha_quic:.8e}")
    print(f"abs(alpha diff): {abs(alpha_quic - alpha_sklearn):.8e}")

    print("\n=== Precision matrix agreement ===")
    print(f"||P_quic - P_sklearn||_F: {diff_fro:.8e}")
    print(f"relative Frobenius diff: {rel_fro:.8e}")
