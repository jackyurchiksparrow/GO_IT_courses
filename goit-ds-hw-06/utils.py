from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from scipy import stats


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
        print("No missing values are found in the dataframe")
        return []

    df_result = pd.DataFrame(rows, columns=["Feature", "NA Count", "Total Count", "NA Percentage"])
    df_result = df_result.sort_values(by="NA Count", ascending=False, ignore_index=True)

    return df_result


def target_info(dataframe: pd.DataFrame, target: str, transform: Literal["log", "log1p"] | None = None):
    if target not in dataframe.columns:
        print(f"{target} is not in df!")
        return

    print("Unique target values:")
    target_col = dataframe[target]
    target_values = pd.DataFrame(
        {
            "counts": target_col.value_counts().to_numpy(),
            "%": np.round(target_col.value_counts(normalize=True).values, 2),
        },
        index=target_col.value_counts().index,
    )

    if target_col.isna().any():
        target_values.loc["NaN"] = [target_col.isna().sum(), np.round(target_col.isna().sum() / len(dataframe) * 100, 2)]

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


def plot_features_against_target(dataframe: pd.DataFrame, continuous_features: list, target: str, imputed_masks: dict | None = None):
    for feat in continuous_features:
        print(f"Generating diagnostic profiling for: {feat}")
        fig, axes = plt.subplots(ncols=5, figsize=(25, 6))

        # --- Plot 1 & 2: Raw Data ---
        sns.histplot(dataframe[feat], kde=True, ax=axes[0], color="#1f77b4")
        stats.probplot(dataframe[feat], plot=axes[1])
        axes[0].set_title("Raw Distribution")
        axes[1].set_title("Raw QQ Plot")

        # Group Header 1
        fig.text(0.21, 0.96, "1. Is this feature normally distributed?", fontsize=12, fontweight="bold", ha="center", color="#2c3e50")

        # --- Plot 3 & 4: Log Transformed Data ---
        sns.histplot(np.log1p(dataframe[feat]), kde=True, ax=axes[2], color="#2ca02c")
        stats.probplot(np.log1p(dataframe[feat]), plot=axes[3])
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
                plot_df = dataframe[[feat]].copy()
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
                plot_df = dataframe[[feat, target]].copy()
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
                data=dataframe,
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
