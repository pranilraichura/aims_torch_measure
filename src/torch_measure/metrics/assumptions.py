# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Assumption checks for G-theory variance-component analyses."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


def _cell_decomposition(
    response_matrix: pd.DataFrame,
    subject_col: str,
    item_col: str,
    trial_col: str,
    response_col: str,
) -> dict:
    """Compute cell means, marginals, and the four families of residuals."""
    import pandas as pd

    required = {subject_col, item_col, trial_col, response_col}
    missing = required - set(response_matrix.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}.")

    df = response_matrix[[subject_col, item_col, trial_col, response_col]].dropna(subset=[response_col])
    if not pd.api.types.is_numeric_dtype(df[response_col]):
        raise ValueError(f"{response_col!r} column must be numeric.")

    n_p = int(df[subject_col].nunique())
    n_i = int(df[item_col].nunique())
    if n_p < 2 or n_i < 2:
        raise ValueError(f"Need at least 2 subjects and 2 items; got n_subjects={n_p}, n_items={n_i}.")

    cell = df.groupby([subject_col, item_col])[response_col].agg(["mean", "count"]).reset_index()
    cell = cell.rename(columns={"mean": "_cell_mean", "count": "_cell_count"})
    if len(cell) < n_p * n_i:
        raise ValueError(
            f"Unbalanced design: {len(cell)}/{n_p * n_i} cells observed. "
            f"Every (subject, item) cell must have at least one observation."
        )

    cell_table = cell.pivot(index=subject_col, columns=item_col, values="_cell_mean").to_numpy(dtype=float)
    grand_mean = cell_table.mean()
    subj_mean = cell_table.mean(axis=1)
    item_mean = cell_table.mean(axis=0)

    has_replications = bool((cell["_cell_count"].to_numpy() > 1).any())

    return {
        "df": df,
        "cell": cell,
        "cell_table": cell_table,
        "grand_mean": grand_mean,
        "subj_mean": subj_mean,
        "item_mean": item_mean,
        "n_p": n_p,
        "n_i": n_i,
        "has_replications": has_replications,
    }


def normality_check(
    response_matrix: pd.DataFrame,
    target: str = "residual",
    subject_col: str = "subject_id",
    item_col: str = "item_id",
    trial_col: str = "trial",
    response_col: str = "response",
) -> dict:
    """Shapiro-Wilk + Q-Q diagnostic on a chosen G-theory effect family.

    Reports the statistic, p-value, and the data needed to draw a Q-Q plot.
    P-values from Shapiro-Wilk are sample-size sensitive (under-powered at
    small N, over-sensitive at large N); inspect the Q-Q output before
    treating any verdict as definitive.

    Parameters
    ----------
    response_matrix : pandas.DataFrame
        Long-form responses, same schema as
        :func:`torch_measure.metrics.variance_components`.
    target : {"residual", "subject_item", "subject", "item"}
        Which effect family to test. ``"residual"`` (within-cell errors)
        requires the design to have replications.
    subject_col, item_col, trial_col, response_col : str
        Column names.

    Returns
    -------
    dict
        Keys: ``test`` (``"shapiro_wilk"``), ``target``, ``statistic`` (W),
        ``p_value``, ``n``, ``qq_theoretical`` (numpy.ndarray), and
        ``qq_sample`` (numpy.ndarray, sorted).
    """
    from scipy import stats

    parts = _cell_decomposition(response_matrix, subject_col, item_col, trial_col, response_col)
    grand_mean = parts["grand_mean"]
    subj_mean = parts["subj_mean"]
    item_mean = parts["item_mean"]
    cell_table = parts["cell_table"]

    if target == "subject":
        values = subj_mean - grand_mean
    elif target == "item":
        values = item_mean - grand_mean
    elif target == "subject_item":
        # Centered cell residuals = interaction + within-cell averaged noise.
        values = (cell_table - subj_mean[:, None] - item_mean[None, :] + grand_mean).reshape(-1)
    elif target == "residual":
        if not parts["has_replications"]:
            raise ValueError(
                "target='residual' requires replications (n_reps > 1 in at least one cell); "
                "use target='subject_item' for a single-observation design."
            )
        df = parts["df"]
        merged = df.merge(parts["cell"][[subject_col, item_col, "_cell_mean"]], on=[subject_col, item_col])
        values = (merged[response_col] - merged["_cell_mean"]).to_numpy(dtype=float)
    else:
        raise ValueError(f"Unknown target: {target!r}. Expected one of subject, item, subject_item, residual.")

    values = np.asarray(values, dtype=float).ravel()
    n = int(values.size)
    if n < 3:
        raise ValueError(f"Need at least 3 observations to run Shapiro-Wilk; got {n}.")

    w_stat, p_value = stats.shapiro(values)
    sorted_sample = np.sort(values)
    theoretical = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)

    return {
        "test": "shapiro_wilk",
        "target": target,
        "statistic": float(w_stat),
        "p_value": float(p_value),
        "n": n,
        "qq_theoretical": theoretical,
        "qq_sample": sorted_sample.astype(float),
    }


def tukey_nonadditivity_test(
    response_matrix: pd.DataFrame,
    subject_col: str = "subject_id",
    item_col: str = "item_id",
    trial_col: str = "trial",
    response_col: str = "response",
) -> dict:
    """Tukey's 1-df test for non-additivity on cell means.

    Especially load-bearing when the design has ``n_reps == 1``: the
    subject x item interaction and the within-cell residual are then
    confounded in :func:`variance_components`, and this test asks whether
    that combined residual contains structured interaction beyond noise.

    Parameters
    ----------
    response_matrix : pandas.DataFrame
        Long-form responses; aggregated to cell means internally.
    subject_col, item_col, trial_col, response_col : str
        Column names.

    Returns
    -------
    dict
        Keys: ``test`` (``"tukey_nonadditivity"``), ``statistic`` (F),
        ``p_value``, ``df_num`` (=1), ``df_den``, ``ss_nonadditivity``,
        ``ss_remainder``.
    """
    from scipy import stats

    parts = _cell_decomposition(response_matrix, subject_col, item_col, trial_col, response_col)
    cell_table = parts["cell_table"]
    grand_mean = parts["grand_mean"]
    a = parts["subj_mean"] - grand_mean  # centered subject effects
    b = parts["item_mean"] - grand_mean  # centered item effects
    n_p = parts["n_p"]
    n_i = parts["n_i"]

    interaction = cell_table - parts["subj_mean"][:, None] - parts["item_mean"][None, :] + grand_mean
    ss_interaction = float(np.sum(interaction**2))

    sum_a_sq = float(np.sum(a**2))
    sum_b_sq = float(np.sum(b**2))
    if sum_a_sq < 1e-12 or sum_b_sq < 1e-12:
        raise ValueError("Subject or item main effect has near-zero variance; Tukey's test is undefined.")

    d = float(np.sum(cell_table * a[:, None] * b[None, :]))
    ss_nonadd = d * d / (sum_a_sq * sum_b_sq)
    ss_remainder = ss_interaction - ss_nonadd

    df_den = (n_p - 1) * (n_i - 1) - 1
    if df_den < 1:
        raise ValueError(
            f"Insufficient residual df for Tukey's test (df_den={df_den}). Need (n_subjects - 1)(n_items - 1) > 1."
        )

    f_stat = ss_nonadd / (ss_remainder / df_den) if ss_remainder > 1e-12 else float("inf")
    p_value = float(1.0 - stats.f.cdf(f_stat, 1, df_den)) if np.isfinite(f_stat) else 0.0

    return {
        "test": "tukey_nonadditivity",
        "statistic": float(f_stat),
        "p_value": p_value,
        "df_num": 1,
        "df_den": int(df_den),
        "ss_nonadditivity": float(ss_nonadd),
        "ss_remainder": float(ss_remainder),
    }


def levene_homogeneity_test(
    response_matrix: pd.DataFrame,
    group_by: str = "item",
    subject_col: str = "subject_id",
    item_col: str = "item_id",
    trial_col: str = "trial",
    response_col: str = "response",
    center: str = "median",
) -> dict:
    """Levene's test for homogeneity of variance across items or subjects.

    G-theory's residual term assumes constant within-cell variance. Levene
    tests whether the variance of the response differs across the chosen
    grouping. Uses the median-centered (Brown-Forsythe) variant by default
    for robustness.

    Parameters
    ----------
    response_matrix : pandas.DataFrame
        Long-form responses; same schema as
        :func:`torch_measure.metrics.variance_components`.
    group_by : {"item", "subject"}
        Which axis to test variance constancy across.
    subject_col, item_col, trial_col, response_col : str
        Column names.
    center : {"median", "mean", "trimmed"}
        Centering for Levene's test; forwarded to ``scipy.stats.levene``.
        ``"median"`` (default) is the Brown-Forsythe variant.

    Returns
    -------
    dict
        Keys: ``test`` (``"levene"``), ``group_by``, ``center``,
        ``statistic`` (W), ``p_value``, ``n_groups``, ``group_sizes``
        (list[int]), ``group_variances`` (list[float], ddof=1).
    """
    from scipy import stats

    if group_by not in {"item", "subject"}:
        raise ValueError(f"group_by must be 'item' or 'subject'; got {group_by!r}.")
    if center not in {"median", "mean", "trimmed"}:
        raise ValueError(f"center must be 'median', 'mean', or 'trimmed'; got {center!r}.")

    parts = _cell_decomposition(response_matrix, subject_col, item_col, trial_col, response_col)
    df = parts["df"]
    key_col = item_col if group_by == "item" else subject_col

    groups: list[np.ndarray] = []
    sizes: list[int] = []
    variances: list[float] = []
    for _, block in df.groupby(key_col, sort=True):
        values = block[response_col].to_numpy(dtype=float)
        if values.size < 2:
            continue
        groups.append(values)
        sizes.append(int(values.size))
        variances.append(float(np.var(values, ddof=1)))

    if len(groups) < 2:
        raise ValueError(f"Need at least 2 groups with >= 2 observations to run Levene's test; got {len(groups)}.")

    w_stat, p_value = stats.levene(*groups, center=center)

    return {
        "test": "levene",
        "group_by": group_by,
        "center": center,
        "statistic": float(w_stat),
        "p_value": float(p_value),
        "n_groups": len(groups),
        "group_sizes": sizes,
        "group_variances": variances,
    }
