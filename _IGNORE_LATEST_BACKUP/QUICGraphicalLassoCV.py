from __future__ import annotations

import operator
import time
import warnings
from numbers import Real
from typing import Any, Tuple, Optional

import numpy as np
from joblib import Parallel, delayed
from sklearn.covariance import EmpiricalCovariance, empirical_covariance, log_likelihood
from sklearn.covariance._graph_lasso import BaseGraphicalLasso, alpha_max  # sklearn 1.4.2
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import check_cv, cross_val_score
from sklearn.utils.validation import _is_arraylike_not_scalar, check_scalar
from sklearn.base import _fit_context


def _center_like_sklearn(X: np.ndarray, assume_centered: bool) -> np.ndarray:
    """Match sklearn empirical_covariance centering semantics."""
    X = np.asarray(X, dtype=float)
    if assume_centered:
        return X
    return X - X.mean(axis=0, keepdims=True)


def _quic_solve_from_data(
    X: np.ndarray,
    alpha: float,
    *,
    quic_pybind,
    max_iter: int,
    verbose: int,
    diag_penalty: float,
    assume_centered: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calls your binding exactly as requested:
        iC, C = quic_pybind.quic(Y_array, lambda_ij, max_iter, L_ii, verbose)

    Where Y_array is p x n (features x samples).
    """
    Xc = _center_like_sklearn(X, assume_centered=assume_centered)  # X is n x p


    Y = np.array(np.ascontiguousarray(Xc.T, dtype=np.float64), order='F')

    iC, C = quic_pybind.QUIC(
        Y,
        float(alpha),          # L_ij
        float(tol),            # tol   <-- required
        int(max_iter),         # max_iter
        float(diag_penalty),   # L_ii  (set to 0.0 for sklearn compatibility)
        int(verbose),
    )

    iC = np.asarray(iC, dtype=float)
    C = np.asarray(C, dtype=float)
    return C, iC  # return (cov, precision) to mirror sklearn path conventions


def _quic_path(
    X: np.ndarray,
    *,
    alphas: np.ndarray,
    quic_pybind,
    X_test: Optional[np.ndarray],
    tol: float,       # kept for signature parity; QUIC may ignore
    enet_tol: float,  # kept for signature parity; QUIC may ignore
    max_iter: int,
    verbose: int,
    diag_penalty: float,
    assume_centered: bool,
):
    """
    Drop-in analogue of sklearn.graphical_lasso_path but solved by QUIC via quic_pybind.quic().
    Returns (covariances, precisions, scores) if X_test is not None; else (covariances, precisions).
    """
    _ = tol, enet_tol  # API parity

    covariances_ = []
    precisions_ = []
    scores_ = []

    test_emp_cov = None
    if X_test is not None:
        # EXACT sklearn call used inside graphical_lasso_path
        test_emp_cov = empirical_covariance(X_test)

    for alpha in alphas:
        try:
            cov, prec = _quic_solve_from_data(
                X,
                float(alpha),
                quic_pybind=quic_pybind,
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

        except Exception:
            covariances_.append(np.nan)
            precisions_.append(np.nan)
            if test_emp_cov is not None:
                scores_.append(-np.inf)

        if verbose == 1:
            import sys
            sys.stderr.write(".")
            sys.stderr.flush()

    if X_test is not None:
        return covariances_, precisions_, scores_
    return covariances_, precisions_


class QUICGraphicalLassoCV(BaseGraphicalLasso):
    """
    GraphicalLassoCV-equivalent CV loop (sklearn 1.4.2 logic), but the solver is QUIC via
        iC, C = quic_pybind.quic(Y, lambda_ij, max_iter, L_ii, verbose)

    Critical equivalence constraint:
      - sklearn GraphicalLassoCV penalizes OFF-DIAGONAL only: alpha * ||Theta||_{1,off}
      - Therefore for strict equivalence set diag_penalty (L_ii) = 0.0 (default).
    """

    def __init__(
        self,
        *,
        quic_pybind,
        alphas=4,
        n_refinements=4,
        cv=None,
        tol=1e-4,
        enet_tol=1e-4,
        max_iter=100,
        mode="cd",          # kept for API parity
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
        self.quic_pybind = quic_pybind
        self.alphas = alphas
        self.n_refinements = n_refinements
        self.cv = cv
        self.n_jobs = n_jobs
        self.diag_penalty = diag_penalty

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        X = self._validate_data(X, ensure_min_features=2)

        # Same location_ semantics as sklearn
        if self.assume_centered:
            self.location_ = np.zeros(X.shape[1])
        else:
            self.location_ = X.mean(axis=0)

        emp_cov = empirical_covariance(X, assume_centered=self.assume_centered)
        cv = check_cv(self.cv, y, classifier=False)

        n_alphas = self.alphas
        inner_verbose = max(0, int(self.verbose) - 1) if isinstance(self.verbose, (int, np.integer)) else int(bool(self.verbose))

        if _is_arraylike_not_scalar(n_alphas):
            for a in self.alphas:
                check_scalar(a, "alpha", Real, min_val=0, max_val=np.inf, include_boundaries="right")
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

                this_path = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                    delayed(_quic_path)(
                        X[train],
                        alphas=alphas,
                        X_test=X[test],
                        quic_pybind=self.quic_pybind,
                        tol=self.tol,
                        enet_tol=self.enet_tol,
                        max_iter=int(0.1 * self.max_iter),  # exact sklearn CV budget
                        verbose=inner_verbose,
                        diag_penalty=float(self.diag_penalty),
                        assume_centered=self.assume_centered,
                    )
                    for train, test in cv.split(X, y)
                )

            covs, _, scores = zip(*this_path)
            covs = zip(*covs)
            scores = zip(*scores)
            path.extend(zip(alphas, scores, covs))
            path = sorted(path, key=operator.itemgetter(0), reverse=True)

            best_score = -np.inf
            last_finite_idx = 0
            for index, (a, sc, _) in enumerate(path):
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

        # build cv_results_ (exact sklearn structure)
        path = list(zip(*path))
        grid_scores = list(path[1])
        alphas_list = list(path[0])

        # add alpha=0 baseline (exact sklearn behavior)
        alphas_list.append(0.0)
        grid_scores.append(
            cross_val_score(
                EmpiricalCovariance(),
                X,
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
            ec = EmpiricalCovariance(assume_centered=self.assume_centered).fit(X)
            self.covariance_ = ec.covariance_
            self.precision_ = ec.precision_
            self.n_iter_ = 0
            self.costs_ = None
            return self

        cov, prec = _quic_solve_from_data(
            X,
            self.alpha_,
            quic_pybind=self.quic_pybind,
            max_iter=self.max_iter,
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
    import numpy as np
    from sklearn.covariance import GraphicalLassoCV
    from sklearn.model_selection import KFold

    # --- import your pybind module ---
    try:
        import quic_pybind
    except Exception as e:
        raise SystemExit(
            f"Failed to import quic_pybind. Build/rename your extension so "
            f"`import quic_pybind` works.\nOriginal error: {e}"
        )

    # Reproducibility
    rng = np.random.default_rng(123)

    # Problem size
    n_samples = 300
    p = 25

    # --- generate a sparse SPD precision matrix (Theta_true) ---
    mask = rng.random((p, p)) < 0.08
    W = rng.uniform(-0.25, 0.25, size=(p, p)) * mask
    W = np.triu(W, 1)
    W = W + W.T  # symmetric off-diagonal

    Theta_true = W.copy()
    # Make diagonally dominant => SPD (practical for a test)
    diag = np.sum(np.abs(Theta_true), axis=1) + 0.5
    np.fill_diagonal(Theta_true, diag)

    # Covariance and samples
    Sigma_true = np.linalg.inv(Theta_true)
    L = np.linalg.cholesky(Sigma_true)
    X = (rng.standard_normal((n_samples, p)) @ L.T).astype(np.float64)  # n x p

    # Use same CV splitter for both models (important)
    cv = KFold(n_splits=5, shuffle=True, random_state=123)

    # Keep settings aligned
    tol = 1e-4
    max_iter = 100
    alphas = 4
    n_refinements = 4

    # --- Fit sklearn GraphicalLassoCV ---
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
    gl.fit(X)

    # --- Fit QUICGraphicalLassoCV (your class above) ---
    qgl = QUICGraphicalLassoCV(
        quic_pybind=quic_pybind,
        alphas=alphas,
        n_refinements=n_refinements,
        cv=cv,
        tol=tol,
        max_iter=max_iter,
        n_jobs=-1,
        verbose=False,
        assume_centered=False,
        diag_penalty=0.0,  # MUST be 0.0 to match sklearn GraphicalLassoCV penalty semantics
    )
    qgl.fit(X)

    # --- Compare ---
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

    # Optional: assert-style guard (comment out if you just want printing)
    # NOTE: exact equality is not guaranteed due to solver differences/tolerances.
    # if abs(alpha_quic - alpha_sklearn) > 1e-8:
    #     print("\nWARNING: alpha differs (this can happen if QUIC solver objective/centering differs).")