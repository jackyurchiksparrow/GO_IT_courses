import inspect
import warnings
from collections.abc import Callable
from functools import wraps
from itertools import combinations
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from IPython.display import display
from scipy import stats
from scipy.linalg import qr as scipy_qr
from sklearn.base import clone
from statsmodels.api import add_constant
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan, het_white
from statsmodels.tsa.stattools import adfuller, kpss


def ensure_target_not_included(func):
    """A Decorator that dynamically drops the target variable from ANY pandas DataFrame passed to the function."""

    @wraps(func)  # to preserve original function's metadata
    def wrapper(*args, **kwargs):  # accept required params and pass all the others
        # Bind the provided arguments to the function's signature
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        # The function must have a target_name argument for us to know what to drop
        target_name = bound_args.arguments.get("target_name")
        if not target_name:
            msg = f"Function '{func.__name__}' must accept a 'target_name' argument to use this decorator."
            raise ValueError(msg)

        # Scan all arguments. If it's a DataFrame, drop the target.
        for name, value in bound_args.arguments.items():
            if isinstance(value, pd.DataFrame) and target_name in value.columns:
                bound_args.arguments[name] = value.drop(columns=[target_name])

        # Execute the original function with the cleaned arguments
        return func(*bound_args.args, **bound_args.kwargs)

    return wrapper


def ensure_categoricals_encoded(func):
    """A Decorator that ensures all categoricals are handled in ALL DataFrames passed to the function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Bind the provided arguments to the function's signature
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        # Scan all arguments for DataFrames and validate their dtypes
        for name, value in bound_args.arguments.items():
            if isinstance(value, pd.DataFrame):
                has_objects = (value.dtypes == "object").any()
                has_categories = (value.dtypes == "category").any()

                if has_objects or has_categories:
                    msg = f"DataFrame argument '{name}' contains unencoded categorical values. Aborted."
                    raise ValueError(msg)

        # Execute if all DataFrames pass the check
        return func(*bound_args.args, **bound_args.kwargs)

    return wrapper


def ensure_no_nulls(func):
    """A Decorator that ensures all nulls are imputed in ALL DataFrames passed to the function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Bind the provided arguments to the function's signature
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        # Scan all arguments for DataFrames and check for missing values
        for name, value in bound_args.arguments.items():
            if isinstance(value, pd.DataFrame) and value.isna().any().any():
                msg = f"DataFrame argument '{name}' has missing values. Aborted."
                raise ValueError(msg)

        # Execute if all DataFrames pass the check
        return func(*bound_args.args, **bound_args.kwargs)

    return wrapper


def ensure_constant_included(func):
    """A Decorator that ensures the column of ones is in @."""

    @wraps(func)
    def wrapper(features: np.ndarray[np.float64], *args, **kwargs):
        if not np.any(np.all(features == 1.0, axis=0)):
            raise ValueError("x_features_with_added_constant must contain a column of ones.")
        return func(features, *args, **kwargs)

    return wrapper


def ensure_series_has_no_nulls(func):
    """A Decorator that ensures a pandas Series argument has no missing values."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        for name, value in bound_args.arguments.items():
            if isinstance(value, pd.Series) and value.isna().any():
                msg = f"Series argument '{name}' has missing values. Aborted."
                raise ValueError(msg)

        return func(*bound_args.args, **bound_args.kwargs)

    return wrapper


def get_null_info(data_frame: pd.DataFrame) -> pd.DataFrame:
    """Renders null info for the specified data frame."""
    feature_columns = data_frame.columns
    rows = []  # to gather missing value metrics

    for feat in feature_columns:
        na_count = data_frame[feat].isna().sum()

        # if any nulls, compute the missing percentage and save
        if na_count > 0:
            total_count = data_frame.shape[0]
            na_percent = na_count / total_count * 100
            rows.append((feat, na_count, total_count, na_percent))

    # if no nulls
    if len(rows) == 0:
        print("No missing values are found in the data_frame")
        return pd.DataFrame()

    # structure ans sort
    df_result = pd.DataFrame(rows, columns=["Feature", "NA Count", "Total Count", "NA Percentage"])
    df_result = df_result.sort_values(by="NA Count", ascending=False, ignore_index=True)

    return df_result


def target_info(data_frame: pd.DataFrame, target: str, transform: Literal["log", "log1p"] | None = None):
    if target not in data_frame.columns:
        print(f"{target} is not in df!")
        return

    print("Unique target values:")
    target_col = data_frame[target]

    target_values = pd.DataFrame(
        {
            "counts": target_col.value_counts().to_numpy(),
            "%": np.round(target_col.value_counts(normalize=True).values, 2),
        },
        index=target_col.value_counts().index,
    )

    if target_col.isna().any():
        target_values.loc["NaN"] = [target_col.isna().sum(), np.round(target_col.isna().sum() / len(data_frame) * 100, 2)]

    display(target_values)

    # 1. Check transformation and setup layout
    if transform is None:
        ncols = 1
        figsize = (10, 4)
    elif transform in ["log", "log1p"]:
        ncols = 2
        figsize = (20, 4)
    else:
        print("Invalid 3rd param:", transform)
        return

    # 2. Create the subplots dynamically
    fig, ax = plt.subplots(ncols=ncols, nrows=1, figsize=figsize)

    # If there is only 1 column, ax is a single object, not a list
    ax_first = ax[0] if ncols == 2 else ax

    # 3. Plot the first graph
    sns.histplot(target_col, kde=True, color="lightcoral", ax=ax_first)
    ax_first.set_title(f"Distribution of {target}")
    ax_first.set_xlabel("Target Value")
    ax_first.set_ylabel("Density")

    # 4. Plot the second graph only if needed
    if transform in ["log", "log1p"]:
        if transform == "log":
            transformed_target = np.log(target_col)
            title_label = f"log({target})"
        else:
            transformed_target = np.log1p(target_col)
            title_label = f"log1p({target})"

        sns.histplot(transformed_target, kde=True, color="lightcoral", ax=ax[1])
        ax[1].set_title(f"Distribution of {title_label}")
        ax[1].set_xlabel("Target Value")
        ax[1].set_ylabel("Density")

    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# ATOMIC PLOT HELPERS — each draws ONE thing onto a provided ax
# ─────────────────────────────────────────────────────────────────────────────


def plot_hist(series: pd.Series, ax, bins="auto", color="#1f77b4", kde=True, title=None):
    """Histogram (+ optional KDE) of a single series onto ax."""
    sns.histplot(series, kde=kde, ax=ax, bins=bins, color=color)
    if title:
        ax.set_title(title)


def plot_qq(series: pd.Series, ax, title=None):
    """QQ plot against a normal distribution onto ax."""
    stats.probplot(series, plot=ax)
    if title:
        ax.set_title(title)


def plot_scatter(data_frame: pd.DataFrame, x: str, y: str, ax, mask: dict | None = None, regline: bool = True, title: str | None = None):
    """Scatter of x vs y onto ax. If mask given (bool Series), masked points are red. regline draws an OLS fit over the UNMASKED points only."""
    if mask is not None:
        plot_df = data_frame[[x, y]].copy()
        plot_df["__mask"] = mask.astype(bool)
        sns.scatterplot(
            data=plot_df,
            x=x,
            y=y,
            hue="__mask",
            palette={False: "#1f77b4", True: "red"},
            ax=ax,
            s=30,
            alpha=0.4,
            legend=False,
            edgecolor="white",
            linewidths=0.5,
        )
        if regline:
            sns.regplot(data=plot_df[~plot_df["__mask"]], x=x, y=y, scatter=False, color="blue", ax=ax)
    else:
        sns.regplot(
            data=data_frame,
            x=x,
            y=y,
            ax=ax,
            scatter=True,
            scatter_kws={"s": 25, "edgecolor": "white", "linewidths": 0.5, "alpha": 0.5},
            line_kws={"color": "blue", "linewidth": 2} if regline else None,
            fit_reg=regline,
        )
    if title:
        ax.set_title(title)


def plot_discrete_vs_target(data_frame: pd.DataFrame, feat, target_name, ax, mask=None):
    """Strip + box plot for a discrete feature against a continuous target."""
    if mask is not None:
        plot_df = data_frame[[feat, target_name]].copy()
        plot_df["__mask"] = mask.astype(bool)
        sns.stripplot(
            data=plot_df,
            x=feat,
            y=target_name,
            hue="__mask",
            palette={False: "#1f77b4", True: "red"},
            ax=ax,
            alpha=0.4,
            jitter=True,
            legend=False,
        )
    else:
        sns.stripplot(data=data_frame, x=feat, y=target_name, color="#1f77b4", alpha=0.3, jitter=True, ax=ax)
        sns.boxplot(
            data=data_frame,
            x=feat,
            y=target_name,
            ax=ax,
            color="white",
            showfliers=False,
            width=0.3,
            medianprops={"color": "#e74c3c", "linewidth": 2.5},
        )
        medians = data_frame.groupby(feat)[target_name].median().sort_index()
        ax.plot(range(len(medians)), medians.values, color="#e74c3c", linewidth=2, marker="o", zorder=5)


def plot_distribution_diagnostics(series: pd.Series, axes, color="#1f77b4", bins="auto", kde=True, label=""):
    """Two-panel univariate diagnostic: histogram + QQ plot. Pass axes as a length-2 sequence."""
    plot_hist(series, axes[0], bins=bins, color=color, kde=kde, title=f"{label} Distribution".strip())
    plot_qq(series, axes[1], title=f"{label} QQ Plot".strip())


def plot_features_against_target(
    data_frame: pd.DataFrame, features: list, target_name: str, bins: dict, imputed_masks: dict | None = None, discrete_threshold: int = 15
):
    """
    Generates a diagnostic grid for features mapped against a target variable.

    Dynamically adjusts plotting structures based on whether a feature is continuous or discrete.
    Continuous features display a 6-plot grid covering raw distributions, log-transformations, and target relationships.
    Discrete features display a condensed 3-plot distribution and trend layout.

    Args:
        data_frame (pd.DataFrame): The source dataset containing features and target columns.
        features (list): Target column names to analyze sequentially.
        target_name (str): Name of the target variable column.
        bins (dict): Dictionary mapping feature names to their specific histogram bin counts
            (e.g., {'feature_name': 30}).
        imputed_masks (dict | None): Optional dictionary of boolean series where True highlights
            imputed or outlier indices across data fields. Defaults to None.
        discrete_threshold (int): Unique value count below which a feature is categorized
            and plotted as discrete. Defaults to 15.

    """
    for i, feat in enumerate(features):
        is_discrete = data_frame[feat].nunique() < discrete_threshold
        n_cols = 3 if is_discrete else 6
        fig_width = 15 if is_discrete else 30
        fig, axes = plt.subplots(ncols=n_cols, figsize=(fig_width, 6))
        bin_number = bins.get(feat) or "auto"

        # Resolve the combined mask once (feature outliers OR target outliers)
        mask = None
        if imputed_masks is not None:
            target_mask = imputed_masks.get(target_name, pd.Series(False, index=data_frame.index))
            mask = (imputed_masks[feat] | target_mask).astype(bool)

        # --- Panels 1-2: raw distribution diagnostics ---
        plot_distribution_diagnostics(data_frame[feat], axes[0:2], color="#1f77b4", bins=bin_number, kde=not is_discrete, label="Raw")

        # --- Panel 3: raw feature vs target ---
        if is_discrete:
            plot_discrete_vs_target(data_frame, feat, target_name, axes[2], mask=mask)
        else:
            plot_scatter(data_frame, feat, target_name, axes[2], mask=mask, regline=True, title="Raw Feature vs Target")

        header_1_x = 0.5 if is_discrete else 0.26
        fig.text(
            header_1_x,
            0.96,
            f"{i}. Raw Feature Diagnostics ({feat}; {'discrete' if is_discrete else 'continuous'})",
            fontsize=14,
            fontweight="bold",
            ha="center",
            color="#2c3e50",
        )

        # --- Panels 4-6: log-transformed (continuous only) ---
        if not is_discrete:
            log_series = np.log1p(data_frame[feat])
            plot_distribution_diagnostics(log_series, axes[3:5], color="#2ca02c", bins=bin_number, kde=True, label="Log(1+x)")

            log_df = data_frame[[feat, target_name]].copy()
            log_df["log_feat"] = log_series
            plot_scatter(log_df, "log_feat", target_name, axes[5], mask=mask, regline=True, title="Log Feature vs Target")

            fig.text(0.76, 0.96, f"{i}. Log(1+x) Feature Diagnostics", fontsize=14, fontweight="bold", ha="center", color="#2c3e50")

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.show()


def plot_box(series, ax, color="#1f77b4", title=None):
    """Vertical boxplot of a single series onto ax."""
    sns.boxplot(y=series, ax=ax, color=color, width=0.4, medianprops={"color": "#e74c3c", "linewidth": 2.5})
    if title:
        ax.set_title(title)


def plot_feature_diagnostics(
    data_frame,
    features,
    bins: dict | None = None,
    transform: Callable | None = None,
    transform_label: str = "Transformed",
    discrete_threshold: int = 15,
):
    """
    Unsupervised per-feature diagnostics (NO target).

    For each feature draws: histogram + QQ + boxplot.
    If `transform` is given (e.g. np.log1p), draws a second row with the
    same three plots on the transformed feature for side-by-side comparison.

    Args:
        data_frame: the data.
        features: list of columns to inspect.
        bins: optional {feature: n_bins}. Missing keys fall back to 'auto'.
        transform: optional callable applied to the feature (e.g. np.log1p, np.sqrt). If None, only the raw row is drawn.
        transform_label: label shown on the transformed row's titles.
        discrete_threshold: features with fewer unique values are treated as discrete (KDE disabled, boxplot still shown).

    """
    bins = bins or {}

    for i, feat in enumerate(features):
        series = data_frame[feat]
        is_discrete = series.nunique() < discrete_threshold
        bin_number = bins.get(feat) or "auto"

        n_rows = 2 if transform is not None else 1
        fig, axes = plt.subplots(nrows=n_rows, ncols=3, figsize=(18, 5 * n_rows))
        # Normalize axes to 2D indexing regardless of row count
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        # ---- Row 1: raw ----
        plot_hist(series, axes[0, 0], bins=bin_number, color="#1f77b4", kde=not is_discrete, title="Raw Distribution")
        plot_qq(series, axes[0, 1], title="Raw QQ Plot")
        plot_box(series, axes[0, 2], color="#1f77b4", title="Raw Boxplot")

        # ---- Row 2: transformed (optional) ----
        if transform is not None:
            t_series = transform(series)
            plot_hist(t_series, axes[1, 0], bins=bin_number, color="#2ca02c", kde=not is_discrete, title=f"{transform_label} Distribution")
            plot_qq(t_series, axes[1, 1], title=f"{transform_label} QQ Plot")
            plot_box(t_series, axes[1, 2], color="#2ca02c", title=f"{transform_label} Boxplot")

        fig.suptitle(
            f"{i}. Feature Diagnostics ({feat}; {'discrete' if is_discrete else 'continuous'})",
            fontsize=14,
            fontweight="bold",
            color="#2c3e50",
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()


@ensure_target_not_included
@ensure_categoricals_encoded
@ensure_no_nulls
def detect_perfect_multicollinearity_via_rank(data_frame: pd.DataFrame, target_name: str) -> int:
    """
    Detects EXACT perfect multicollinearity using QR decomposition with column pivoting. Reports only true algebraic dependencies (feature = linear combination of other features), NOT high-but-imperfect correlations.

    Args:
        data_frame (pd.DataFrame): data frame; doesn't need to be standardized, it does it anyway, but no problem if it is
        target_name (str): target name

    Returns:
        matrix rank

    """
    X = data_frame.copy()
    mean = X.mean()
    std = X.std()
    std[std == 0] = 1.0  # Avoid division by zero for constants
    X_std = (X - mean) / std

    _, R, _ = scipy_qr(X_std.to_numpy(), pivoting=True)
    diag = np.abs(np.diag(R))
    tol = diag[0] * 1e-10
    rank = int(np.sum(diag > tol))
    n_features = X.shape[1]

    print(f"Matrix rank: {rank} / {n_features} features")
    if rank < n_features:
        print(f"Perfect multicollinearity detected — {n_features - rank} exact dependencies.")
    return rank


# tries to find the smallest possible subset of features to drop that resolves perfect multicollinearity in one go; prioritizes feature-engineered predictors
# (i.e., sorts by moving features with underscore to the top; I tend to call derived predictors according to that standard)
@ensure_target_not_included
@ensure_no_nulls
@ensure_categoricals_encoded
def detect_features_to_restore_full_rank(data_frame: pd.DataFrame, target_name: str, prioritize_dropping_by: str = "_"):
    """
    Identifies minimal set of features to drop in order to restore full column rank (i.e., eliminate perfect multicollinearity).

    - Computes initial rank of feature matrix (excluding target; correlation between features and the target is expected and does not cause multicollinearity).
    - If rank < n_features → perfect multicollinearity exists.
    - Iterates through feature subsets to find smallest set to drop that restores independence.
    - Prioritizes dropping features that include @prioritize_dropping_by in their respective col names

    Args:
        data_frame (pd.DataFrame): df with fully encoded categoricals, no nulls, no target
        target_name (str): target name
        prioritize_dropping_by (str): a substring found in @data_frame's col names to prioritize dropping them

    Returns:
        list[str] | str:
            - Names of features to drop if solution exists.
            - "No perfect multicollinearity detected" if none present.
            - "Could not resolve..." if no subset fixes rank.

    """
    features = data_frame.columns.to_list()

    # Prioritize dropping features containing the specific substring
    features.sort(key=lambda x: prioritize_dropping_by not in x)

    initial_matrix = data_frame.copy()
    initial_rank = detect_perfect_multicollinearity_via_rank(initial_matrix, target_name)
    n_features = len(features)

    print("--- Starting Multicollinearity Check ---")
    print(f"Total Features: {n_features}")
    print(f"Initial Matrix Rank: {initial_rank}")

    if initial_rank == n_features:
        print("Result: Matrix is already full rank!")
        return "No perfect multicollinearity detected"

    # Calculate exactly how many features need to be dropped total
    needed_drops = n_features - initial_rank
    print(f"Target: Must drop at least {needed_drops} feature(s) to restore full rank.\n")

    # Track combinations that successfully fix the rank deficit
    successful_drops = []

    # Try combinations starting from the minimum needed drops up to all features
    for k in range(needed_drops, n_features + 1):
        print(f"--> Checking combinations of size {k}...")

        # Count total combinations in this step for the progress print
        all_combos = list(combinations(features, k))
        total_combos = len(all_combos)

        for index, combo in enumerate(all_combos, 1):
            # Print progress update every 100 combinations (or change to 1 for smaller lists)
            if index % 100 == 0 or index == total_combos:
                print(f"    Progress: {index}/{total_combos} combinations checked...", end="\r")

            candidate_features = [f for f in features if f not in combo]
            candidate_rank = detect_perfect_multicollinearity_via_rank(data_frame[candidate_features], target_name)

            # If the remaining features are fully independent, we found a match!
            if candidate_rank == len(candidate_features):
                successful_drops.append(list(combo))

                print(f"\n[FOUND] Dropping {list(combo)} successfully restores full rank!")
                print(f"New Rank: {candidate_rank} matches remaining feature count ({len(candidate_features)}).")

                # Since we want the SMALLEST set, we can return immediately once this size 'k' finishes
                return list(combo)

    print("\nResult: Failed to resolve matrix rank.")
    return "Could not resolve perfect multicollinearity by dropping any subset of features."


@ensure_target_not_included
@ensure_categoricals_encoded
@ensure_no_nulls
def perform_advanced_vif_analysis(
    data_frame: pd.DataFrame,
    target_name: str,
    to_scale: bool = False,
) -> pd.DataFrame:
    """
    Performs VIF analysis + evaluates the condition matrix number (kappa) on the given data_frame using the specified predictors.

    Includes the calculation of theta (Standard Error Inflation Factor) to account for sample size buffer.

    Args:
        data_frame (pd.DataFrame): df with fully encoded categoricals, no nulls
        target_name (str): target name
        to_scale (bool): if True, standardizes the data before analysis

    Returns:
        a data_frame with pairs {feature: its VIF, theta, matrix_kappa}

    """
    # Scale data (StandardScaler equivalent) if requested to prevent non-essential collinearity
    if to_scale:
        data_frame = (data_frame - data_frame.mean()) / data_frame.std()

    predictors = data_frame.columns.to_numpy()
    n = len(data_frame)

    # Fast VIF Calculation via Correlation Matrix Inverse
    corr_matrix = data_frame.corr().to_numpy()
    try:
        inv_corr = np.linalg.inv(corr_matrix)
        # np.abs safeguards against floating-point precision issues causing tiny negative numbers
        vif_values = np.abs(np.diag(inv_corr))
    except np.linalg.LinAlgError:
        # Handles perfect multicollinearity where matrix cannot be inverted
        return pd.DataFrame({"feature": predictors, "VIF": np.inf, "theta": np.inf, "matrix_kappa": "-"})

    # 1.5 Calculate SEIF (Standard Error Inflation Factor / Theta)
    # Ratio of variance inflation to sample size buffer: sqrt(VIF) / sqrt(n - 1)
    theta_values = np.sqrt(vif_values) / np.sqrt(n - 1)

    # 2. Parallel Diagnostic: Compute Condition Index of the Matrix
    # Standardize data first to avoid scale distortion (skip redundant scaling if to_scale is True)
    x_scaled = data_frame if to_scale else (data_frame - data_frame.mean()) / data_frame.std()

    # Add constant after scaling so it doesn't get zeroed out
    x_scaled = add_constant(x_scaled, prepend=False)

    # SVD on design matrix to get singular values
    singular_values = np.linalg.svd(x_scaled.to_numpy(), compute_uv=False)

    # Kappa (Condition Number) is the ratio of max to min singular value
    # Filter out near-zero singular values to avoid zero-division errors
    min_sv = np.min(singular_values)
    matrix_kappa = (np.max(singular_values) / min_sv) if min_sv > 1e-10 else np.inf

    # 3. Construct the output DataFrame cleanly
    vif_df = pd.DataFrame(
        {
            "feature": predictors,
            "VIF": vif_values,
            "theta": theta_values,  # Tracks mathematical stability (must be < 0.20)
            "matrix_kappa": matrix_kappa,  # Broadcasts the single matrix score to all rows
        },
    )

    return vif_df.sort_values(by="VIF", ascending=False).reset_index(drop=True)


@ensure_target_not_included
@ensure_categoricals_encoded
@ensure_no_nulls
def get_vif_features_to_drop(
    data_frame: pd.DataFrame,
    target_name: str,
    theta_threshold: float = 0.20,
    to_scale: bool = False,
) -> list[str]:
    """
    Iteratively identifies and suggests predictors to drop to reduce multicollinearity until the mathematically safe Standard Error Inflation Factor (theta) threshold is reached.

    (1) Runs advanced VIF analysis via `perform_advanced_vif_analysis()`.
    (2) Iteratively removes the feature with the highest theta (which also has the highest VIF).
    (3) Records the max VIF, theta, and matrix condition number (kappa) after each drop.

    Args:
        data_frame (pd.DataFrame): The DataFrame containing encoded categoricals and no nulls.
        target_name (str): The name of the target variable.
        theta_threshold (float, optional): SEIF (theta) safety threshold to stop dropping. Defaults to 0.20.
        to_scale (bool): If True, standardizes the data before performing matrix calculations.

    Returns:
        list[str]: Steps showing max VIF, max theta, and matrix kappa before/after each drop.

    """
    # 1. Baseline calculation
    vif_df = perform_advanced_vif_analysis(data_frame, target_name, to_scale=to_scale)

    max_idx = vif_df["theta"].idxmax()
    max_theta = vif_df.loc[max_idx, "theta"]
    max_vif = vif_df.loc[max_idx, "VIF"]
    kappa_val = vif_df["matrix_kappa"].iloc[0]

    # Helper to format kappa safely (handles "-" for perfect collinearity without ValueError)
    def format_kappa(k):
        return f"{k:.4f}" if isinstance(k, (int, float)) and k != float("inf") else str(k)

    # We use a 30-character left-aligned layout for the action text to keep numbers aligned
    to_drop = [
        f"{'before dropping anything:':<30} VIF = {max_vif:<10.4f} theta = {max_theta:<8.4f} kappa = {format_kappa(kappa_val)}",
    ]

    # 2. Evaluate against the mathematical safety pillar (theta)
    while max_theta >= theta_threshold:
        # Locate the name of the feature column holding the maximum score
        feat = vif_df.loc[max_idx, "feature"]

        # Drop the problematic col
        data_frame = data_frame.drop(columns=[feat])

        # Recalculate VIF, theta, and kappa on the reduced feature subset
        vif_df = perform_advanced_vif_analysis(data_frame, target_name, to_scale=to_scale)

        # Update the max values for the NEXT loop condition AND for the print statement
        max_idx = vif_df["theta"].idxmax()
        max_theta = vif_df.loc[max_idx, "theta"]
        max_vif = vif_df.loc[max_idx, "VIF"]
        kappa_val = vif_df["matrix_kappa"].iloc[0]

        action_text = f"after dropping {feat}:"
        to_drop.append(
            f"{action_text:<30} VIF = {max_vif:<10.4f} theta = {max_theta:<8.4f} kappa = {format_kappa(kappa_val)}",
        )

    return to_drop


def render_correlations(data_frame: pd.DataFrame, is_to_plot: bool = True, threshold: float = 0.5) -> pd.Series:
    """
    Finds feature pairs with a Pearson correlation coefficient above a given threshold. Analyzes linear relationships, ignoring autocorrelation and pair duplication.

    Optionally plots Pearson's correlations where |corr| >= @threshold.

    Args:
        data_frame (pd.DataFrame): categoricals and nulls automatically skipped by the library, target MUST be numerical.
        is_to_plot (bool): whether to plot the heatmap
        threshold (float): filter only correlations >= threshold (both for plot and return)

    Returns:
        (pd.Series) Filtered correlations with the threshold

    """
    corr_matrix = data_frame.corr()

    # Створюємо маску для верхнього трикутника матриці (виключаючи головну діагональ)
    # k=1 означає зсув вище діагоналі
    upper_mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)

    # Застосовуємо маску: все інше перетвориться на NaN
    upper_tri = corr_matrix.where(upper_mask)

    # stack() стискає матрицю в Series, автоматично видаляючи NaN
    flat_pairs = upper_tri.stack().reset_index()
    flat_pairs.columns = ["Feature 1", "Feature 2", "Correlation"]

    # Фільтруємо за абсолютним значенням порогу (враховуємо і -0.9, і +0.9)
    filtered_pairs = flat_pairs[flat_pairs["Correlation"].abs() >= threshold]

    # if no entries with the @threshold
    if filtered_pairs.empty:
        print(f"No entries to plot for a threshold of {threshold}")
    else:
        if is_to_plot:
            # 1. Збираємо унікальні назви фіч, які перевищили поріг
            high_corr_features = list(set(filtered_pairs["Feature 1"]).union(set(filtered_pairs["Feature 2"])))

            # 2. Фільтруємо оригінальну 2D матрицю за цими фічами
            matrix_to_plot = corr_matrix.loc[high_corr_features, high_corr_features]

            # 3. Будуємо графік за валідною 2D матрицею
            plt.figure(figsize=(12, 10))
            sns.heatmap(
                matrix_to_plot,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                vmin=-1,
                vmax=1,
                linewidths=0.5,
            )
            plt.title(f"Feature correlations Matrix (|r| >= {threshold})")
            plt.tight_layout()
            plt.show()

    # Сортуємо за спаданням абсолютної кореляції
    return filtered_pairs.sort_values(
        by="Correlation",
        key=abs,
        ascending=False,
    ).reset_index(drop=True)


def get_cooks_distances(data_frame_train: pd.DataFrame, fit_model) -> tuple[pd.Series, pd.Series]:
    """
    Computes cook's distances for each datapoint.

    Args:
        data_frame_train (pd.DataFrame): Original data (to get index alignment).
        fit_model (statsmodels fitted model): Fitted OLS or GLM with `.get_influence()` available.

    Returns:
        tuple:
            - cooks (pd.Series): {Row index: Cook's D value} (sorted descending).
            - flagged (pd.Series): {Row index: Cook's D value} exceeding 4/n threshold.

    """
    # Influence measures
    influence = fit_model.get_influence()
    cooks_d, cooks_p = influence.cooks_distance

    # Put into DataFrame for inspection
    cooks = pd.Series(cooks_d, index=data_frame_train.index, name="cooks_d").sort_values(ascending=False)
    top = cooks.head(20)

    # thresholds
    n = len(data_frame_train)
    threshold_4n = 4.0 / n
    print(f"\n4/n threshold = {threshold_4n:.6g}")
    print("Count above 4/n (baseline anomalies):", (cooks > threshold_4n).sum())
    print("Count above 1.0 (severe anomalies):", (cooks > 1.0).sum())

    print("\nTop 20 Cook's D values (descending):")
    print(top)

    # return flagged indices
    return cooks, cooks[cooks > threshold_4n]


@ensure_constant_included
def breusch_pagan_test(x_features_with_added_constant: np.ndarray[np.float64], residuals: np.ndarray[np.float64]) -> pd.DataFrame:
    """Runs the Breusch-Pagan test to assess heteroscedasticity."""
    lm_stat, lm_pval, f_stat, f_pval = het_breuschpagan(residuals, x_features_with_added_constant)

    if lm_pval >= 0.05:
        print("Fail to reject H0: No Heteroscedasticity detected.")
    else:
        print("Reject H0: Heteroscedasticity is present.")

    return pd.DataFrame(
        [{"LM Statistic": lm_stat, "R2": lm_stat / len(residuals), "LM p-value": lm_pval, "F-value": f_stat, "F p-value": f_pval}],
    ).style.hide(axis="index")


@ensure_constant_included
def white_test(x_features_with_added_constant: np.ndarray[np.float64], residuals: np.ndarray[np.float64]) -> pd.DataFrame:
    """Runs the Breusch-Pagan test to assess heteroscedasticity."""
    lm_stat, lm_pval, f_stat, f_pval = het_white(residuals, x_features_with_added_constant)

    if lm_pval >= 0.05:
        print("Fail to reject H0: No Heteroscedasticity detected.")
    else:
        print("Reject H0: Heteroscedasticity is present.")

    return pd.DataFrame(
        [{"LM Statistic": lm_stat, "R2": lm_stat / len(residuals), "LM p-value": lm_pval, "F-value": f_stat, "F p-value": f_pval}],
    ).style.hide(axis="index")


@ensure_target_not_included
@ensure_no_nulls
@ensure_categoricals_encoded
def find_redundant_columns_qr(df: pd.DataFrame, target_name: str) -> list:
    """
    Identifies linearly dependent (redundant) columns using QR decomposition with pivoting.

    This function detects multi-collinearity by standardizing the features and
    performing a pivoted QR decomposition (X * P = Q * R). Columns that are linearly
    dependent on prior columns are pivoted to the back of the matrix, where their
    corresponding diagonal entries in the upper triangular matrix R drop near zero.

    Args:
        df (pd.DataFrame): df_train containing both features and the target.
        target_name (str): The name of the target column.

    Returns:
        list: A list of feature names identified as linearly dependent/redundant and safe to drop.

    Raises:
        ValueError: If features have zero variance (causing division by zero during standardization).

    """
    tol = 1e-10
    X = df.copy()
    cols = X.columns.tolist()

    # Standardize — raw feature scales differ by orders of magnitude which
    # causes the SVD tolerance to misclassify near-zero singular values
    X_std = (X - X.mean()) / X.std()

    _, R, P = scipy_qr(X_std.values, pivoting=True)

    # Diagonal of R: near-zero entries signal linearly dependent columns
    diag = np.abs(np.diag(R))
    rank = int(np.sum(diag > tol * diag[0]))
    redundant_indices = P[rank:]  # column indices pivoted to the back = redundant

    redundant_cols = [cols[i] for i in redundant_indices]

    print(f"Rank: {rank} / {len(cols)}")
    print(f"Redundant columns ({len(redundant_cols)}): {redundant_cols}")
    return redundant_cols


@ensure_no_nulls
@ensure_categoricals_encoded
@ensure_target_not_included
def calculate_feature_psi(x_sample: pd.DataFrame, x_evaluation: pd.DataFrame, target_name: str, num_bins: int = 10) -> pd.Series:
    """
    Computes Population Stability Index between two dataframes.

    The function ensures the categoricals are encoded, but they don't have to be (numerically).
    They are only required to be valid: no typos, consistent across both @x_sample and @x_evaluation

    Args:
        x_sample (np.ndarray): 2D baseline data to analyze
        x_evaluation (np.ndarray): 2D true population sample (production/live data or the data on which the train will perform)
        target_name (str): the dependent variable name in @x_sample and @x_evaluation
        num_bins (int): determined based on sample size

    Returns:
        pd.Series: A series containing the PSI score for each continuous feature.

    """
    psi_results = {}

    # Ensure we only compare columns that exist in both datasets
    common_cols = [col for col in x_sample.columns if col in x_evaluation.columns]

    for col in common_cols:
        sample_arr = x_sample[col].to_numpy()
        eval_arr = x_evaluation[col].to_numpy()

        # Define bin edges based strictly on sample quantiles
        percentiles = np.linspace(0, 100, num_bins + 1)
        bin_edges = np.percentile(sample_arr, percentiles)

        # Adjust boundaries to handle edge values safely
        bin_edges[0] -= 1e-5
        bin_edges[-1] += 1e-5

        # Calculate counts per bin
        base_counts, _ = np.histogram(sample_arr, bins=bin_edges)
        eval_counts, _ = np.histogram(eval_arr, bins=bin_edges)

        # Convert counts to proportions (percentages)
        expected = base_counts / len(sample_arr)
        actual = eval_counts / len(eval_arr)

        # Rule to prevent Division by Zero / Log of Zero for empty bins
        expected = np.where(expected == 0, 1e-4, expected)
        actual = np.where(actual == 0, 1e-4, actual)

        # Compute PSI formula component
        psi_value = np.sum((actual - expected) * np.log(actual / expected))
        psi_results[col] = float(psi_value)

    return pd.Series(psi_results, name="PSI_Score").sort_values(ascending=False)


def bootstrap_coefficients(
    data_frame: pd.DataFrame,  # cleaned DataFrame (train), rows in chronological order
    target_name: str,  # target column name (string)
    predictors: np.ndarray,  # list of predictor column names (strings) - same columns used in model
    n_boot: int = 500,  # number of bootstrap samples
    block_len: int = 24,  # 'None' will run -> i.i.d. bootstrap (plain), otherwise MBB block length in rows (e.g., 24 for hourly daily)
    random_state: int = 42,
):
    """
    Estimates stability of regression coefficients via bootstrap (i.i.d. or moving-block).

    - If block_len=None: i.i.d. bootstrap (sample rows).
    - If block_len > 0: Moving-block bootstrap (sample blocks of consecutive rows).
    - Fits OLS on each resample.
    - Collects coefficient estimates.
    - Reports mean, std, relative std, sign consistency, and 95% CI.

    Args:
        data_frame (pd.DataFrame): Cleaned training data (chronological order).
        target_name (str): Target variable name.
        predictors (np.ndarray | list[str]): Predictor variable names.
        n_boot (int, default=500): Number of bootstrap replicates.
        block_len (int, default=24): Block length for moving-block bootstrap.
                                     Use None for i.i.d. resampling.
        random_state (int, default=42): RNG seed.

    Returns:
        pd.DataFrame (index=features):
            - coef_mean: Mean bootstrapped coefficient.
            - coef_std: Standard deviation.
            - relative_std: Std normalized by mean magnitude.
            - sign_consistency: Proportion of bootstraps with same sign.
            - ci_low, ci_high: 95% percentile confidence intervals.

    """
    if target_name in predictors:
        raise ValueError("Target mustn't be present in predictors")

    rng = np.random.RandomState(random_state)
    n = len(data_frame)
    cols = ["const", *list(predictors)]
    collected = []

    if n < 30:
        raise ValueError("Data too small for meaningful bootstrap. Need more rows.")
    if block_len is not None and block_len <= 0:
        block_len = None

    # Build overlapping blocks for moving-block bootstrap if requested
    blocks = None
    n_blocks_needed = None
    if block_len is not None:
        if block_len >= n:
            # fallback to i.i.d. if block_len too large
            block_len = None
        else:
            blocks = []
            for start in range(0, n - block_len + 1):
                blocks.append(data_frame.iloc[start : start + block_len].reset_index(drop=True))
            n_blocks_needed = int(np.ceil(n / block_len))

    for i in range(n_boot):
        if block_len is None:
            # i.i.d. bootstrap: sample indices with replacement
            idx = rng.choice(n, size=n, replace=True)
            sample = data_frame.iloc[idx].reset_index(drop=True)
        else:
            # moving-block bootstrap: sample blocks with replacement and concat
            chosen = [blocks[rng.randint(0, len(blocks))] for _ in range(n_blocks_needed)]  # select the built block 'n_blocks_needed' times
            sample = (
                pd.concat(chosen, ignore_index=True).iloc[:n].reset_index(drop=True)
            )  # concat the current selected set of blocks into a data_frame

        # drop rows with NaNs for safety (rare if 'data_frame' cleaned)
        sample = sample[[target_name, *list(predictors)]].dropna()
        if sample.shape[0] < max(30, len(predictors) + 5):
            # if resample leaves too few rows, skip
            continue

        try:
            Xs = sm.add_constant(sample[predictors])
            ys = sample[target_name]
            model = sm.OLS(ys, Xs).fit()
            params_series = model.params
            params_dict = params_series.to_dict()
            params_vec = np.array([params_dict.get(c, 0.0) for c in cols], dtype=float)
            collected.append(params_vec)
        except Exception as err:
            raise ValueError(err) from err
            continue

        if (i + 1) % 200 == 0:
            print(f"{i + 1}/{n_boot} bootstraps completed...")

    if len(collected) == 0:
        raise RuntimeError("All bootstrap fits failed — check data / predictors.")

    arr = np.vstack(collected)  # shape (n_effective_boot, n_features)
    mean = arr.mean(axis=0)
    sd = arr.std(axis=0, ddof=1)

    # relative std (std normalized by the feature's mean)
    eps = 1e-12
    denom = np.where(np.abs(mean) < eps, eps, np.abs(mean))  # if denominator is < 1e-12, we set it to eps=1e-12 to avoid 0 division
    rel_std = sd / denom

    # how often the sign changes from initial means
    mean_sign = np.sign(mean)
    sign_consistency = (np.sign(arr) == mean_sign).mean(axis=0)

    # 95% percentile boundaries
    ci_low = np.percentile(arr, 2.5, axis=0)
    ci_high = np.percentile(arr, 97.5, axis=0)

    out = pd.DataFrame(
        {
            "feature": cols,
            "coef_mean": mean,
            "coef_std": sd,
            "relative_std": rel_std,
            "sign_consistency": sign_consistency,
            "ci_low": ci_low,
            "ci_high": ci_high,
        },
    )

    out = out.set_index("feature")
    return out


@ensure_categoricals_encoded
@ensure_no_nulls
@ensure_target_not_included
def auto_iqr_winsorization_limits(df: pd.DataFrame, target_name: str, k: float = 1.5) -> tuple[dict, pd.DataFrame]:
    """
    Calculates Tukey's fence outlier boundaries and flags values for winsorization.

    It calculates winsorization boundaries with IQR automatically, then logs the expected impact per feature and builds a global outlier mask.

    Args:
        df (pd.DataFrame): training df.
        target_name (str): The name of the target column.
        k (float, default=1.5): The multiplier for the IQR to determine the outlier fences. A value of 1.5 defines standard outliers, while 3.0 defines extreme outliers.

    Returns:
        tuple[dict, pd.DataFrame]: A tuple containing:
            - winsor_limits (dict): Dictionary mapping feature names to a tuple of their calculated
              clipping boundaries: {feature_name: (lower_bound, upper_bound)}.
            - winsorized_masks (pd.DataFrame): A boolean DataFrame of identical dimensions to the
              input features, where True signifies that the specific value is an outlier that
              would be capped.

    """
    features = df.columns
    winsor_limits = {}

    # Ініціалізуємо датафрейм масок з False (такої ж розмірності, як df[features])
    winsorized_masks = pd.DataFrame(False, index=df.index, columns=features)

    for feat in features:
        X = df[feat]

        # Розраховуємо квартилі та IQR
        q25 = X.quantile(0.25)
        q75 = X.quantile(0.75)
        iqr = q75 - q25

        # Визначаємо математичні межі паркану Тьюкі
        lower_bound = q25 - (k * iqr)
        upper_bound = q75 + (k * iqr)

        # Захист: межі не повинні виходити за реальні min/max
        lower_bound = max(lower_bound, X.min())
        upper_bound = min(upper_bound, X.max())

        # Записуємо ліміти
        winsor_limits[feat] = (lower_bound, upper_bound)

        # Створюємо булеву маску ДІЙСНИХ викидів, які вийшли за розрахований паркан
        is_lower_outlier = X < lower_bound
        is_upper_outlier = X > upper_bound

        # Зберігаємо в загальний датафрейм масок
        winsorized_masks[feat] = is_lower_outlier | is_upper_outlier

        # Рахуємо реальну кількість точок, які будуть кліпнуті
        lower_sum = is_lower_outlier.sum()
        upper_sum = is_upper_outlier.sum()
        total_outliers = lower_sum + upper_sum

        print(
            f"{feat:<10}: {total_outliers / len(X):.2%} ({upper_sum} upper and {lower_sum} lower) "
            f"of {feat} would be capped (Limits: [{lower_bound:.4f}, {upper_bound:.4f}])",
        )

    return winsor_limits, winsorized_masks


@ensure_series_has_no_nulls
def check_stationarity(
    series: pd.Series,
    target_name: str = None,
    alpha: float = 0.05,
    window: int = None,  # leave at default, if the rolling mean looks jagged, increase it; if it looks flat regardless of what you do, the feature probably has no drift and you can skip the visual check
    plot: bool = True,
    verbose: bool = True,
):
    """
    Combined ADF + KPSS stationarity check with a 4-quadrant verdict.

    - ADF  H0: series HAS a unit root (non-stationary). Reject (p < alpha) -> stationary.
    - KPSS H0: series IS stationary.                     Reject (p < alpha) -> non-stationary.

    Both are still p-value-based tests and share the general large-N caveat (statistically
    significant != practically meaningful with enough rows). Treat the verdict as a fast
    screen, not a final answer -- when ADF and KPSS disagree, or even when they agree,
    trust the rolling mean/std plot (plot=True) over either p-value; it's the diagnostic
    that doesn't depend on sample size.

    Args:
        series (pd.Series): Time series to test (e.g. the target, hourly-indexed).
        target_name (str): Optional, used only for print/plot labeling.
        alpha (float, default=0.05): Significance level for both tests.
        window (int): Rolling window for the mean/std plot. Defaults to a small fraction
            of series length -- override with something tied to your data's natural cycle
            (e.g. 24 for hourly data with a daily cycle).
        plot (bool, default=True): Draw raw series + rolling mean/std.
        verbose (bool, default=True): Print results to console.

    Returns:
        dict: {'adf': {...}, 'kpss': {...}, 'verdict': str, 'agreement': bool}

    """
    s = series.dropna().astype(float)
    n = len(s)
    if n < 10:
        raise ValueError("Series too short for stationarity testing (need more observations).")

    # --- ADF ---
    adf_stat, adf_pvalue, adf_usedlag, adf_nobs, adf_crit = adfuller(s.values, regression="c", autolag="AIC")[:5]
    adf_is_stationary = adf_pvalue < alpha

    # --- KPSS ---
    with warnings.catch_warnings():
        # statsmodels caps KPSS p-values to its lookup-table range (e.g. "p-value is
        # smaller than the smallest p-value tested") -- silenced here for readability,
        # but the printed p-value can be pinned at 0.01 or 0.10 rather than exact.
        warnings.simplefilter("ignore")
        kpss_stat, kpss_pvalue, kpss_usedlag, kpss_crit = kpss(s.values, regression="c", nlags="auto")
    kpss_is_stationary = kpss_pvalue >= alpha  # opposite direction from ADF

    # --- Combined verdict ---
    if adf_is_stationary and kpss_is_stationary:
        verdict, agreement = "stationary", True
    elif not adf_is_stationary and not kpss_is_stationary:
        verdict, agreement = "non-stationary", True
    else:
        verdict, agreement = "conflicting", False

    if verbose:
        label = f" [{target_name}]" if target_name else ""
        print(f"--- Stationarity check{label} (n={n}) ---")
        print(
            f"ADF  stat={adf_stat:.4f}  p={adf_pvalue:.4f}  used_lag={adf_usedlag}  "
            f"-> {'stationary' if adf_is_stationary else 'non-stationary'}",
        )
        print(
            f"KPSS stat={kpss_stat:.4f}  p={kpss_pvalue:.4f}  used_lag={kpss_usedlag}  "
            f"-> {'stationary' if kpss_is_stationary else 'non-stationary'}",
        )
        print(f"=> Verdict: {verdict.upper()}" + ("" if agreement else " -- tests disagree, check the plot before deciding."))

    if plot:
        w = window or min(max(n // 10, 2), 30)
        roll_mean, roll_std = s.rolling(w).mean(), s.rolling(w).std()

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(s.values, alpha=0.3, label="raw")
        ax.plot(roll_mean.values, label=f"rolling mean (w={w})")
        ax.plot(roll_std.values, label=f"rolling std (w={w})")
        ax.set_title(f"Stationarity{(' — ' + target_name) if target_name else ''} — verdict: {verdict}")
        ax.legend()
        plt.show()

    return {
        "adf": {
            "stat": adf_stat,
            "p_value": adf_pvalue,
            "used_lag": adf_usedlag,
            "nobs": adf_nobs,
            "critical_values": adf_crit,
            "is_stationary": adf_is_stationary,
        },
        "kpss": {
            "stat": kpss_stat,
            "p_value": kpss_pvalue,
            "used_lag": kpss_usedlag,
            "critical_values": kpss_crit,
            "is_stationary": kpss_is_stationary,
        },
        "verdict": verdict,
        "agreement": agreement,
    }


def plot_stationarity_diagnostic(data_frame: pd.DataFrame, column_name: str, window_size: int = 30, segment_by: str | None = None):
    """
    Plots the rolling mean and standard deviation of a continuous variable.

    This diagnostic tool visualizes how the mean and variance of a time series
    evolve over time to evaluate its stationarity. It supports classification and regression.

    Args:
        data_frame (pd.DataFrame): The input DataFrame containing the time series and optional grouping metadata (for @segment_by).
        column_name (str): The name of the column containing the continuous numerical values to analyze.
        window_size (int, default=30): The rolling window size used to calculate the moving average and standard deviation.
        segment_by (str, optional): The name of the categorical column used to segment and group the data before plotting. Defaults to None, which analyzes the series globally.

    Returns:
        None: Displays a dual-axis matplotlib line plot.

    """
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax2 = ax1.twinx()

    if segment_by is None:
        s = data_frame[column_name]
        ax1.plot(s.values, color="blue", alpha=0.25, label="raw")
        ax1.plot(s.rolling(window_size).mean().values, color="red", linewidth=2, label=f"rolling mean (w={window_size})")
        ax2.plot(s.rolling(window_size).std().values, color="black", linewidth=1.5, linestyle="--", label="rolling std (right axis)")
    else:
        colors = plt.cm.tab10.colors
        for i, (label, g) in enumerate(data_frame.groupby(segment_by)):
            s = g[column_name].reset_index(drop=True)  # reset so x-axis is contiguous
            c = colors[i % len(colors)]
            ax1.plot(s.values, color=c, alpha=0.15)
            ax1.plot(s.rolling(window_size, min_periods=1).mean().values, color=c, linewidth=2, label=f"mean — class {label}")
            ax2.plot(s.rolling(window_size, min_periods=1).std().values, color=c, linewidth=1, linestyle="--", label=f"std — class {label}")

    ax1.set_ylabel("Rolling mean", fontsize=11)
    ax2.set_ylabel("Rolling std", fontsize=11)
    ax1.set_title(f"Stationarity: {column_name}" + (f"  |  by {segment_by}" if segment_by else ""))

    # combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.show()


@ensure_no_nulls
@ensure_categoricals_encoded
def screen_autocorrelation_stage1(
    df: pd.DataFrame,
    target_name: str,
    max_lag: int = 40,
    alpha: float = 0.05,
    mode: str = "regression",
) -> pd.DataFrame:
    """
    Stage 1: screen raw series for temporal autocorrelation BEFORE fitting any model.

    Goal: identify which features (and target, for regression) have temporal memory worth exploiting via lag features.

    Regression: checks all columns including target as a single chronological series.
    Classification: checks each feature WITHIN each class separately to avoid confusing genuine within-state persistence with class-transition jumps.
    The target (discrete labels) is excluded — Ljung-Box on categorical integers
    would just detect that classes persist over consecutive rows, which is trivially
    known and not actionable.

    Args:
        df: df_train sorted chronologically; with only numerical continuous features
        target_name: Target column name (regardless of whether it is discrete or not)
        max_lag: Joint test covers lags 1 through max_lag.
        alpha: Flagging threshold.
        mode: "regression" or "classification".

    Returns:
        DataFrame of results sorted by p-value ascending (most suspicious first).

    """
    records = []

    if mode == "regression":
        cols_to_check = df.columns  # target included — it's continuous and meaningful to check
        for col in cols_to_check:
            series = df[col].to_numpy()
            lb_test = acorr_ljungbox(series, lags=[max_lag], return_df=True)
            records.append(
                {
                    "Feature": col,
                    "Is Target": col == target_name,
                    f"LB_Stat (Lag {max_lag})": lb_test.loc[max_lag, "lb_stat"],
                    "p-value": lb_test.loc[max_lag, "lb_pvalue"],
                    "Flagged": lb_test.loc[max_lag, "lb_pvalue"] < alpha,
                },
            )

    elif mode == "classification":
        feature_cols = [c for c in df.columns if c != target_name]
        for class_label in df[target_name].unique():
            class_df = df[df[target_name] == class_label]
            for col in feature_cols:
                series = class_df[col].to_numpy()
                if len(series) <= max_lag:
                    # not enough rows in this class to test at this lag depth
                    continue
                lb_test = acorr_ljungbox(series, lags=[max_lag], return_df=True)
                records.append(
                    {
                        "Class": class_label,
                        "Feature": col,
                        f"LB_Stat (Lag {max_lag})": lb_test.loc[max_lag, "lb_stat"],
                        "p-value": lb_test.loc[max_lag, "lb_pvalue"],
                        "Flagged": lb_test.loc[max_lag, "lb_pvalue"] < alpha,
                    },
                )

    return pd.DataFrame(records).sort_values("p-value").reset_index(drop=True)


@ensure_categoricals_encoded
@ensure_no_nulls
def run_acorr_ljungbox_for_classification_residuals(
    df: pd.DataFrame,
    target_name: str,
    class_labels: list,
    model,  # any sklearn-compatible classifier with predict_proba
    max_lag: int = 20,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Evaluates temporal dependency in classifier residuals using a One-vs-Rest strategy.

    For each class label, this function trains a fresh, binary instance of the model
    (the specified class vs. all other classes). It calculates the raw residuals
    (y_true - p_predicted) and runs a joint Ljung-Box test up to `max_lag`. This
    helps determine if the model is failing to exploit temporal structures within
    specific activity states or classes.

    Args:
        df (pd.DataFrame): Training DataFrame sorted chronologically.
        target_name (str): The name of the categorical target/label column.
        class_labels (list): List of unique class labels present in the target column to test individually.
        model (BaseEstimator): An unfitted, scikit-learn compatible classifier object implements the `predict_proba` method.
        max_lag (int, default=20): The maximum lag depth included in the joint Ljung-Box test.
        alpha (float, default=0.05): The significance level used to flag remaining autocorrelation in residuals.

    Returns:
        pd.DataFrame: A summary DataFrame containing one row per class with columns:
            - Class: The class label evaluated.
            - LB_Stat (Lag X): The calculated Ljung-Box test statistic.
            - p-value: The test p-value.
            - Flagged: Boolean indicating if significant residual correlation remains.

    """
    df_copy = df.copy()
    X = df_copy.drop(columns=[target_name])
    if "const" in X.columns:
        X = X.drop(columns=["const"])

    records = []
    for class_label in class_labels:
        y_binary = (df_copy[target_name] == class_label).astype(int)

        # clone() creates a fresh unfitted copy — prevents state leaking between classes
        clf = clone(model)
        clf.fit(X, y_binary)
        p_hat = clf.predict_proba(X)[:, 1]
        resid = y_binary.to_numpy() - p_hat

        lb_test = acorr_ljungbox(resid, lags=[max_lag], return_df=True)
        records.append(
            {
                "Class": class_label,
                f"LB_Stat (Lag {max_lag})": lb_test.loc[max_lag, "lb_stat"],
                "p-value": lb_test.loc[max_lag, "lb_pvalue"],
                "Flagged": lb_test.loc[max_lag, "lb_pvalue"] < alpha,
            },
        )

    return pd.DataFrame(records)


@ensure_no_nulls
@ensure_categoricals_encoded
def run_acorr_ljungbox_for_regression_residuals(
    df: pd.DataFrame,
    target_name: str,
    model,  # any sklearn-compatible regressor with predict
    max_lag: int = 20,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Evaluates temporal dependency in regressor residuals.

    This function fits a fresh instance of the provided regression model on the
    chronological features, computes the raw forecasting errors (y_true - y_pred),
    and applies a joint Ljung-Box test. If the residuals are flagged, it means the
    model has left unexploited time-series signals on the table (e.g., missing lags).

    Args:
        df (pd.DataFrame): Training DataFrame sorted chronologically.
        target_name (str): The name of the continuous numerical target column.
        model (BaseEstimator): An unfitted, scikit-learn compatible regressor object implements the `predict` method.
        max_lag (int, default=20): The maximum lag depth included in the joint Ljung-Box test.
        alpha (float, default=0.05): The significance level used to flag remaining autocorrelation in residuals.

    Returns:
        pd.DataFrame: A single-row DataFrame containing the overall diagnostic results:
            - LB_Stat (Lag X): The calculated Ljung-Box test statistic.
            - p-value: The test p-value.
            - Flagged: Boolean indicating if significant residual correlation remains.

    """
    X = df.drop(columns=[target_name])
    y = df[target_name]

    reg = clone(model)
    reg.fit(X, y)
    resid = y.to_numpy() - reg.predict(X)

    lb_test = acorr_ljungbox(resid, lags=[max_lag], return_df=True)
    return pd.DataFrame(
        [
            {
                f"LB_Stat (Lag {max_lag})": lb_test.loc[max_lag, "lb_stat"],
                "p-value": lb_test.loc[max_lag, "lb_pvalue"],
                "Flagged": lb_test.loc[max_lag, "lb_pvalue"] < alpha,
            },
        ],
    )
