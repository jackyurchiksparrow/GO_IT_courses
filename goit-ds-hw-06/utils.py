from functools import wraps
from itertools import combinations
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from numpy.linalg import matrix_rank
from scipy import stats
from statsmodels.api import add_constant


def ensure_target_not_included(func):
    """A Decorator that drops the @target from the @df if not dropped."""

    @wraps(func)  # to preserve original function's metadata
    def wrapper(df: pd.DataFrame, target_name: str, *args, **kwargs):  # accept required params and pass all the others
        df_copy = df.copy()
        if target_name in df_copy.columns:
            df_copy = df_copy.drop(columns=[target_name])
        return func(df_copy, target_name, *args, **kwargs)

    return wrapper


def ensure_categoricals_encoded(func):
    """A Decorator that ensures all categoricals are handled in @df."""

    @wraps(func)
    def wrapper(df: pd.DataFrame, *args, **kwargs):
        df_copy = df.copy()
        if (df_copy.dtypes == "object").any():
            raise ValueError("Not all categorical values were encoded. Aborted.")
        return func(df_copy, *args, **kwargs)

    return wrapper


def ensure_no_nulls(func):
    """A Decorator that ensures all nulls are imputed in @df."""

    @wraps(func)
    def wrapper(df: pd.DataFrame, *args, **kwargs):
        df_copy = df.copy()
        if df_copy.isna().any().any():
            raise ValueError("Dataframe has missing values. Aborted.")
        return func(df_copy, *args, **kwargs)

    return wrapper


def get_null_info(data_frame: pd.DataFrame) -> pd.DataFrame:
    feature_columns = data_frame.columns
    rows = []  # an empty list to gather missing value metrics for each feature

    for feat in feature_columns:
        # total number of nulls in the current column
        na_count = data_frame[feat].isna().sum()

        # if any nulls, compute the missing percentage and save
        if na_count > 0:
            total_count = data_frame.shape[0]
            na_percent = na_count / total_count * 100
            rows.append((feat, na_count, total_count, na_percent))

    # if no nulls
    if len(rows) == 0:
        print("No missing values are found in the data_frame")
        # Note: Returns an empty list object instead of a DataFrame structure
        return pd.DataFrame()

    # Convert the collected list of tuples into a structured results DataFrame
    df_result = pd.DataFrame(rows, columns=["Feature", "NA Count", "Total Count", "NA Percentage"])

    # Sort the results so features with the highest missing values appear at the top
    df_result = df_result.sort_values(by="NA Count", ascending=False, ignore_index=True)

    # Return the finalized summary report DataFrame
    return df_result


def get_null_info(data_frame: pd.DataFrame) -> pd.DataFrame:
    feature_columns = data_frame.columns
    rows = []

    for feat in feature_columns:
        na_count = data_frame[feat].isna().sum()
        if na_count > 0:
            total_count = data_frame.shape[0]
            na_percent = na_count / total_count * 100

            rows.append((feat, na_count, total_count, na_percent))

    if len(rows) == 0:
        print("No missing values are found in the data_frame")
        return []

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

    _, ax = plt.subplots(ncols=2, nrows=1, figsize=(20, 4))

    sns.histplot(target_col, kde=True, color="lightcoral", ax=ax[0])
    ax[0].set_title(f"Distribution of {target}")
    ax[0].set_xlabel("Target Value")
    ax[0].set_ylabel("Density")

    if transform == "log":
        transformed_target = np.log(target_col)
    elif transform == "log1p":
        transformed_target = np.log1p(target_col)
    elif transform is None:
        return
    else:
        print("Invalid 3rd param:", transform)
        return

    sns.histplot(transformed_target, kde=True, color="lightcoral", ax=ax[1])
    ax[1].set_title(f"Distribution of log({target})")
    ax[1].set_xlabel("Target Value")
    ax[1].set_ylabel("Density")

    plt.show()


def plot_features_against_target(data_frame: pd.DataFrame, continuous_features: list, target: str, imputed_masks: dict | None = None):
    for feat in continuous_features:
        print(f"Generating diagnostic profiling for: {feat}")
        fig, axes = plt.subplots(ncols=5, figsize=(25, 6))

        # --- Plot 1 & 2: Raw Data ---
        sns.histplot(data_frame[feat], kde=True, ax=axes[0], color="#1f77b4")
        stats.probplot(data_frame[feat], plot=axes[1])
        axes[0].set_title("Raw Distribution")
        axes[1].set_title("Raw QQ Plot")

        # Group Header 1
        fig.text(0.21, 0.96, "1. Is this feature normally distributed?", fontsize=12, fontweight="bold", ha="center", color="#2c3e50")

        # --- Plot 3 & 4: Log Transformed Data ---
        sns.histplot(np.log1p(data_frame[feat]), kde=True, ax=axes[2], color="#2ca02c")
        stats.probplot(np.log1p(data_frame[feat]), plot=axes[3])
        axes[2].set_title("Log(1+x) Distribution")
        axes[3].set_title("Log(1+x) QQ Plot")

        # Group Header 2
        fig.text(0.61, 0.96, "2. Does a log transformation fix the skewness?", fontsize=12, fontweight="bold", ha="center", color="#2c3e50")

        # --- Plot 5: Relationship with Target ---
        if imputed_masks is not None:
            feat_mask = imputed_masks[feat]
            target_mask = imputed_masks[target]
            combined_mask = (feat_mask | target_mask).astype(bool)

            if feat == target:
                plot_df = data_frame[[feat]].copy()
                plot_df["mask"] = combined_mask

                sns.scatterplot(
                    x=plot_df[feat], y=plot_df[feat], ax=axes[4], legend=False, s=50, label="Not outlier", marker="o", alpha=0.4
                )
                sns.scatterplot(
                    x=plot_df.loc[plot_df["mask"], feat],
                    y=plot_df.loc[plot_df["mask"], feat],
                    ax=axes[4],
                    legend=False,
                    color="red",
                    s=50,
                    marker="X",
                    label="Outlier",
                )
            else:
                plot_df = data_frame[[feat, target]].copy()
                plot_df["mask"] = combined_mask

                # Draw the scatter points, marking outliers in Red
                sns.scatterplot(
                    data=plot_df,
                    x=feat,
                    y=target,
                    hue="mask",
                    s=30,
                    palette={False: "#3885BC", True: "red"},
                    ax=axes[4],
                    legend=False,
                    edgecolor="white",
                    linewidths=0.5,
                    alpha=0.4,
                )
                # Draw the trend line strictly using the clean points
                sns.regplot(data=plot_df[~plot_df["mask"]], x=feat, y=target, scatter=False, color="blue", ax=axes[4])
        else:
            sns.regplot(
                data=data_frame,
                x=feat,
                y=target,
                ax=axes[4],
                scatter=True,
                scatter_kws={"s": 25, "edgecolor": "white", "linewidths": 0.5, "alpha": 0.5},
                line_kws={"color": "blue", "linewidth": 2},
            )
        axes[4].set_title("Linear Target Trend")

        # Group Header 3
        fig.text(0.9, 0.96, "3. Outliers & Linear Strength", fontsize=12, fontweight="bold", ha="center", color="#2c3e50")

        # Clean up layout margins so text doesn't overlap plots
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.show()


@ensure_target_not_included
@ensure_categoricals_encoded
@ensure_no_nulls
def detect_perfect_multicollinearity_via_rank(data_frame: pd.DataFrame, target_name: str) -> tuple[bool, int]:
    """
    Returns matrix rank (to diagnoses perfect multicollinearity).

    Args:
        data_frame (pd.DataFrame): df with fully encoded categoricals, no nulls
        target_name (str): target name

    """
    # accept only numerical and encoded categoricals
    if (data_frame.dtypes == "object").any():
        raise ValueError("Not all categorical values were encoded. Aborted.")

    rank = matrix_rank(data_frame.to_numpy())
    n_features = data_frame.shape[1]

    print(f"Matrix rank: {rank} / {n_features} features")

    if rank < n_features:
        print("Perfect multicollinearity detected.")
        return True, rank

    return False, rank


# tries to find the smallest possible subset of features to drop that resolves perfect multicollinearity in one go; prioritizes feature-engineered predictors
# (i.e., sorts by moving features with underscore to the top; I tend to call derived predictors according to that standard)
@ensure_target_not_included
@ensure_no_nulls
@ensure_categoricals_encoded
def detect_features_to_restore_full_rank(data_frame: pd.DataFrame, target_name: str, prioritize_dropping_by: str = "_"):
    """
    Identifies minimal set of features to drop in order to restore full column rank
    (i.e., eliminate perfect multicollinearity).

    - Computes initial rank of feature matrix (excluding target).
    - If rank < n_features → perfect multicollinearity exists.
    - Iterates through feature subsets to find smallest set to drop that restores independence.
    - Prioritizes engineered features (features with "_").

    Args:
        data_frame (pd.DataFrame): Input dataset.
        target (str): Target variable (excluded from analysis).

    Returns:
        list[str] | str:
            - Names of features to drop if solution exists.
            - "No perfect multicollinearity detected" if none present.
            - "Could not resolve..." if no subset fixes rank.

    """  # noqa: D205
    features = data_frame.columns.to_numpy()
    features.sort(key=lambda x: "_" not in x)  # prioritize dropping feature-engineered features

    initial_rank = matrix_rank(data_frame.dropna().values)  # drop nulls (just in case) and calculate the rank
    n_features = len(features)

    # If there's no perfect multicollinearity, nothing to drop
    if initial_rank == n_features:
        return "No perfect multicollinearity detected"

    # Try all combinations of 1 up to n features to remove
    for k in range(1, n_features + 1):
        for combo in combinations(features, k):
            # Keep features not in the current combo
            candidate_features = [f for f in features if f not in combo]

            # Check if the remaining features are linearly independent separately
            candidate_rank = matrix_rank(data_frame[candidate_features].values)
            if candidate_rank == len(candidate_features):
                return list(combo)

    # If no subset fixes the issue
    return "Could not resolve perfect multicollinearity by dropping any subset of features."


@ensure_target_not_included
@ensure_categoricals_encoded
@ensure_no_nulls
def perform_advanced_vif_analysis(data_frame: pd.DataFrame, target_name: str, ascending: bool = False) -> pd.DataFrame:
    """
    Performs VIF analysis + evaluates the condition matrix number (kappa) on the given data_frame using the specified predictors.

    Args:
        data_frame (pd.DataFrame): df with fully encoded categoricals, no nulls
        target_name (str): target name
        ascending (bool): sorting by VIF (defaults to descending)

    Returns:
        a data_frame with pairs {feature: its VIF}

    """
    predictors = data_frame.columns.to_numpy()

    # Fast VIF Calculation via Correlation Matrix Inverse
    corr_matrix = data_frame.corr().to_numpy()
    try:
        inv_corr = np.linalg.inv(corr_matrix)
        vif_values = np.diag(inv_corr)
    except np.linalg.LinAlgError:
        # Handles perfect multicollinearity where matrix cannot be inverted
        return pd.DataFrame({"feature": predictors, "VIF": np.inf, "kappa": "-"})

    # 2. Parallel Diagnostic: Compute Condition Index of the Matrix
    # Standardize data first to avoid scale distortion
    x_scaled = (data_frame - data_frame.mean()) / data_frame.std()
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
            "matrix_kappa": matrix_kappa,  # Broadcasts the single matrix score to all rows
        }
    )

    return vif_df.sort_values(by="VIF", ascending=ascending).reset_index(drop=True)


@ensure_target_not_included
@ensure_categoricals_encoded
@ensure_no_nulls
def get_vif_features_to_drop(data_frame: pd.DataFrame, target_name: str, threshold: float = 5.0) -> list[str]:
    """
    Iteratively identifies and suggests predictors to drop to reduce multicollinearity until the specified VIF threshold is reached.

    (1) Runs VIF analysis via `utils.perform_vif()`, (2) iteratively removes the feature with highest VIF ≥ threshold, (3) records max VIF after each drop.

    Args:
        data_frame (pd.DataFrame): The DataFrame containing encoded categoricals and no nulls.
        target_name (str): The name of the target variable.
        threshold (float, optional): VIF threshold to stop dropping. Defaults to 5.0.

    Returns:
        list[str]: Steps showing max VIF before/after each drop.

    """
    # baseline VIF calculation, extract the highest VIF
    vif_df = perform_advanced_vif_analysis(data_frame, target_name)
    max_val = vif_df["VIF"].max()
    to_drop = [f"before dropping anything: {max_val}"]

    # Evaluate against threshold; loops one extra time if a previous drop brought it under
    while max_val >= threshold:
        max_val = vif_df["VIF"].max()

        # Locate the name of the feature column holding the maximum
        feat = vif_df.loc[vif_df["VIF"].idxmax(), "feature"]

        # if the highest score breaks the threshold ceiling
        if max_val >= threshold:
            # Drop the problematic col
            data_frame = data_frame.drop(columns=[feat])

            # Recalculate VIF scores on the reduced feature subset
            vif_df = perform_advanced_vif_analysis(data_frame, target_name)
            to_drop.append(f"after dropping {feat}: {vif_df['VIF'].max()}")
        else:
            return to_drop

    return to_drop


def plot_correlations(data_frame: pd.DataFrame, features_to_plot: list, title: str, threshold: float = 0.0):
    correlations = data_frame[features_to_plot].corr()

    # Find features that have any strong correlation (excluding self-correlation)
    mask = (abs(correlations) >= threshold) & (~np.eye(len(correlations), dtype=bool))
    keep_features = correlations.columns[mask.any(axis=1)]

    # Filter to those features
    filtered = correlations.loc[keep_features, keep_features]

    if filtered.empty:
        print(f"No entries to plot for a threshold of {threshold}")
        return

    np.fill_diagonal(filtered.values, np.nan)

    plt.figure(figsize=(15, 15))
    sns.heatmap(filtered, annot=True, fmt=".2f")

    if threshold != 0.0:
        title = title + f" (|corr| ≥ {threshold})"

    plt.title(title)


def plot_cross_validation_performance(train_loss, val_loss, train_r2, val_r2):
    # Use clean, modern presentation styling
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    _, axes = plt.subplots(ncols=2, figsize=(16, 5.5))
    epochs_range = np.arange(1, len(train_loss) + 1)

    # 1. Plotting Loss Curve Trajectories
    axes[0].plot(epochs_range, train_loss, label="Averaged Train Loss", color="#1f77b4", linewidth=2)
    axes[0].plot(epochs_range, val_loss, label="Averaged Validation Loss", color="#ff7f0e", linewidth=2, linestyle="--")
    axes[0].set_title("Cross-Validated MSE", fontsize=13, fontweight="bold", pad=15)
    axes[0].set_xlabel("Epochs", fontsize=11, labelpad=8)
    axes[0].set_ylabel("Loss Value", fontsize=11, labelpad=8)
    axes[0].legend(frameon=True, facecolor="white", edgecolor="none", fontsize=10)
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # 2. Plotting R2 Metric Trajectories
    axes[1].plot(epochs_range, train_r2, label="Averaged Train R2", color="#2ca02c", linewidth=2)
    axes[1].plot(epochs_range, val_r2, label="Averaged Validation R2", color="#d62728", linewidth=2, linestyle="--")
    axes[1].set_title("Cross-Validated R2", fontsize=13, fontweight="bold", pad=15)
    axes[1].set_xlabel("Epochs", fontsize=11, labelpad=8)
    axes[1].set_ylabel("R2 Score", fontsize=11, labelpad=8)

    # Restrict lower bounds for R2 to prevent massive negative jumps from polluting the chart scale
    axes[1].set_ylim(bottom=max(-1.0, min(np.min(train_r2), np.min(val_r2)) - 0.1), top=1.05)
    axes[1].legend(frameon=True, facecolor="white", edgecolor="none", fontsize=10)
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.show()
