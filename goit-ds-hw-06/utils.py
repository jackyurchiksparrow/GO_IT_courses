import inspect
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
from statsmodels.stats.diagnostic import het_breuschpagan, het_white


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


def plot_features_against_target(data_frame: pd.DataFrame, continuous_features: list, target_name: str, imputed_masks: dict | None = None):
    """
    Generates a 6-plot diagnostic grid for each continuous feature.

    What each plot is used for:
        Plot 1: Raw Distribution (Histogram + KDE). Visualizes the distribution and skewness of the raw feature.
        Plot 2: Raw QQ Plot (Quantile-Quantile Plot). Shows how far the raw feature is from normality.
        Plot 3: Raw Feature vs Target (Scatter Plot + Regression Line). Evaluates the structural trend and
                heteroscedasticity on raw dimensions. Supports outlier masking.
        Plot 4: Log(1+x) Distribution (Histogram + KDE). Displays the feature after a log1p transformation.
        Plot 5: Log(1+x) QQ Plot. Confirms whether the log-transformed feature satisfies normality assumptions.
        Plot 6: Log Feature vs Target (Scatter Plot + Regression Line). Evaluates variance stability and linearity
                after the transformation. Supports outlier masking.

    Args:
        data_frame (pd.DataFrame): The source dataset containing features and target.
        continuous_features (list): String names of continuous columns to analyze.
        target_name (str): Name of the target variable column.
        imputed_masks (dict | None): Optional dictionary of boolean series mapping True to outlier or imputed locations.

    """
    for feat in continuous_features:
        print(f"Generating diagnostic profiling for: {feat}")

        # Expanded to 6 columns to balance Raw vs. Log pipelines evenly
        fig, axes = plt.subplots(ncols=6, figsize=(30, 6))

        # --- Plots 1 & 2: Raw Data (Normality Assessment) ---
        sns.histplot(data_frame[feat], kde=True, ax=axes[0], color="#1f77b4")
        stats.probplot(data_frame[feat], plot=axes[1])
        axes[0].set_title("Raw Distribution")
        axes[1].set_title("Raw QQ Plot")

        # --- Plot 3: Raw Feature vs Target (Outlier Impact & Linearity) ---
        if imputed_masks is not None:
            feat_mask = imputed_masks[feat]
            target_mask = imputed_masks[target_name]
            combined_mask = (feat_mask | target_mask).astype(bool)

            if feat == target_name:
                plot_df = data_frame[[feat]].copy()
                plot_df["mask"] = combined_mask

                sns.scatterplot(x=plot_df[feat], y=plot_df[feat], ax=axes[2], legend=False, s=50, marker="o", alpha=0.4)
                sns.scatterplot(
                    x=plot_df.loc[plot_df["mask"], feat],
                    y=plot_df.loc[plot_df["mask"], feat],
                    ax=axes[2],
                    legend=False,
                    color="red",
                    s=50,
                    marker="X",
                )
            else:
                plot_df = data_frame[[feat, target_name]].copy()
                plot_df["mask"] = combined_mask

                sns.scatterplot(
                    data=plot_df,
                    x=feat,
                    y=target_name,
                    hue="mask",
                    s=30,
                    palette={False: "#1f77b4", True: "red"},
                    ax=axes[2],
                    legend=False,
                    edgecolor="white",
                    linewidths=0.5,
                    alpha=0.4,
                )
                sns.regplot(data=plot_df[~plot_df["mask"]], x=feat, y=target_name, scatter=False, color="blue", ax=axes[2])
        else:
            sns.regplot(
                data=data_frame,
                x=feat,
                y=target_name,
                ax=axes[2],
                scatter=True,
                scatter_kws={"s": 25, "edgecolor": "white", "linewidths": 0.5, "alpha": 0.5},
                line_kws={"color": "blue", "linewidth": 2},
            )
        axes[2].set_title("Raw Feature vs Target")

        # Group Header 1: Centered over the first three plots
        fig.text(0.26, 0.96, "1. Raw Feature Diagnostics", fontsize=14, fontweight="bold", ha="center", color="#2c3e50")

        # --- Plots 4 & 5: Log Transformed Data (Variance Stabilization Check) ---
        log_feat_series = np.log1p(data_frame[feat])
        sns.histplot(log_feat_series, kde=True, ax=axes[3], color="#2ca02c")
        stats.probplot(log_feat_series, plot=axes[4])
        axes[3].set_title("Log(1+x) Distribution")
        axes[4].set_title("Log(1+x) QQ Plot")

        # --- Plot 6: Log Feature vs Target ---
        if imputed_masks is not None:
            if feat == target_name:
                plot_df = data_frame[[feat]].copy()
                plot_df["mask"] = combined_mask
                log_vals = np.log1p(plot_df[feat])

                sns.scatterplot(x=log_vals, y=log_vals, ax=axes[5], legend=False, s=50, marker="o", alpha=0.4)
                sns.scatterplot(
                    x=log_vals.loc[plot_df["mask"]],
                    y=log_vals.loc[plot_df["mask"]],
                    ax=axes[5],
                    legend=False,
                    color="red",
                    s=50,
                    marker="X",
                )
            else:
                plot_df = data_frame[[feat, target_name]].copy()
                plot_df["mask"] = combined_mask
                plot_df["log_feat"] = np.log1p(plot_df[feat])

                sns.scatterplot(
                    data=plot_df,
                    x="log_feat",
                    y=target_name,
                    hue="mask",
                    s=30,
                    palette={False: "#1f77b4", True: "red"},
                    ax=axes[5],
                    legend=False,
                    edgecolor="white",
                    linewidths=0.5,
                    alpha=0.4,
                )
                sns.regplot(data=plot_df[~plot_df["mask"]], x="log_feat", y=target_name, scatter=False, color="blue", ax=axes[5])
        else:
            sns.regplot(
                x=np.log1p(data_frame[feat]),
                y=data_frame[target_name],
                ax=axes[5],
                scatter=True,
                scatter_kws={"s": 25, "edgecolor": "white", "linewidths": 0.5, "alpha": 0.5},
                line_kws={"color": "blue", "linewidth": 2},
            )
        axes[5].set_title("Log Feature vs Target")

        # Group Header 2: Centered over the final three plots
        fig.text(0.76, 0.96, "2. Log(1+x) Feature Diagnostics", fontsize=14, fontweight="bold", ha="center", color="#2c3e50")

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.show()


@ensure_target_not_included
@ensure_categoricals_encoded
@ensure_no_nulls
def detect_perfect_multicollinearity_via_rank(data_frame: pd.DataFrame, target_name: str) -> tuple[bool, int]:
    """
    Returns matrix rank (to diagnoses perfect multicollinearity).

    Args:
        data_frame (pd.DataFrame): df with fully encoded categoricals, no nulls, no target
        target_name (str): target name

    Returns:
        matrix rank

    """
    # accept only numerical and encoded categoricals
    if (data_frame.dtypes == "object").any():
        raise ValueError("Not all categorical values were encoded. Aborted.")

    rank = matrix_rank(data_frame.to_numpy())
    n_features = data_frame.shape[1]

    print(f"Matrix rank: {rank} / {n_features} features")

    if rank < n_features:
        print("Perfect multicollinearity detected.")

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
    features = data_frame.columns.to_numpy()
    features.sort(key=lambda x: prioritize_dropping_by not in x)  # prioritize dropping feature-engineered features

    initial_rank = matrix_rank(data_frame.to_numpy())  # drop nulls (just in case) and calculate the rank
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
        },
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
            to_drop.append(f"after dropping {feat}: VIF = {vif_df['VIF'].max()}, kappa = {vif_df['matrix_kappa']}")
        else:
            return to_drop

    return to_drop


def plot_correlations(data_frame: pd.DataFrame, title: str = "Feature correlations", threshold: float = 0.5):
    """
    Plots correlations where |corr| >= @threshold.

    Args:
        data_frame (pd.DataFrame): categoricals and nulls automatically skipped by the library.
        title (str): plot title
        threshold (float): render only correlations >= threshold

    """
    correlations = data_frame.corr()

    # Find features with |corr| >= @threshold, exclude self-correlations
    mask = (abs(correlations) >= threshold) & (~np.eye(len(correlations), dtype=bool))
    keep_features = correlations.columns[mask.any(axis=1)]
    filtered = correlations.loc[keep_features, keep_features]
    filtered = filtered.mask(np.eye(len(filtered), dtype=bool))

    # if no entries with the @threshold
    if filtered.empty:
        print(f"No entries to plot for a threshold of {threshold}")
        return

    # plot
    plt.figure(figsize=(15, 15))
    sns.heatmap(filtered, annot=True, fmt=".2f")
    plt.title(title)
    plt.show()


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

    return pd.Series(psi_results, name="PSI_Score")
