"""Phase 3: data overview and EDA on the training set."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .table_display import style_colored_table as shared_style_colored_table
from pandas.plotting import scatter_matrix

from .phase_1_import_library import ProjectConfig, display_path_collection


MISSING_TOKENS = {"", "na", "nan", "null", "none", "unknown"}
TABLE_STYLES = [
    {
        "selector": "caption",
        "props": [
            ("caption-side", "top"),
            ("font-weight", "700"),
            ("font-size", "15px"),
            ("color", "#12343b"),
            ("padding", "8px 0"),
            ("text-align", "left"),
        ],
    },
    {
        "selector": "th",
        "props": [
            ("background-color", "#12343b"),
            ("color", "white"),
            ("font-weight", "700"),
            ("border", "1px solid #cbd5e1"),
            ("padding", "8px"),
        ],
    },
    {
        "selector": "td",
        "props": [
            ("border", "1px solid #e5e7eb"),
            ("padding", "8px"),
        ],
    },
    {
        "selector": "tbody tr:nth-child(even)",
        "props": [("background-color", "#f8fafc")],
    },
    {
        "selector": "tbody tr:nth-child(odd)",
        "props": [("background-color", "#ffffff")],
    },
]


def style_colored_table(
    df: pd.DataFrame,
    caption: str,
    precision: int = 3,
    cmap: str = "YlGnBu",
    gradient_columns: list[str] | None = None,
):
    """Return a high-contrast colored pandas Styler for notebook display."""

    return shared_style_colored_table(
        df=df,
        caption=caption,
        precision=precision,
        cmap=cmap,
        gradient_columns=gradient_columns,
    )

def missing_mask(series: pd.Series) -> pd.Series:
    return series.isna() | series.astype(str).str.strip().str.lower().isin(MISSING_TOKENS)


def dataframe_info_text(df: pd.DataFrame) -> str:
    buffer = StringIO()
    df.info(buf=buffer)
    return buffer.getvalue()


def summarize_split(df: pd.DataFrame, split_name: str, target_column: str = "price") -> dict[str, Any]:
    target_exists = target_column in df.columns
    return {
        "split": split_name,
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "target": target_column if target_exists else None,
        "missing_values": {column: int(missing_mask(df[column]).sum()) for column in df.columns},
        "duplicate_rows": int(df.duplicated().sum()),
        "info": dataframe_info_text(df),
    }


def create_data_overview_summary(
    raw_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str = "price",
) -> dict[str, Any]:
    """Return split summaries for notebook display without modifying data."""

    numeric_train_df = train_df.apply(pd.to_numeric, errors="coerce").select_dtypes(include=[np.number])
    correlation_matrix = numeric_train_df.corr(numeric_only=True)
    if target_column in correlation_matrix.columns:
        top_price_correlations = (
            correlation_matrix[target_column]
            .drop(labels=[target_column], errors="ignore")
            .dropna()
            .sort_values(key=lambda series: series.abs(), ascending=False)
            .head(12)
            .rename("correlation_with_price")
            .reset_index()
            .rename(columns={"index": "feature"})
        )
    else:
        top_price_correlations = pd.DataFrame(columns=["feature", "correlation_with_price"])

    return {
        "raw": summarize_split(raw_df, "raw", target_column),
        "train": summarize_split(train_df, "train", target_column),
        "validation": summarize_split(val_df, "validation", target_column),
        "test": summarize_split(test_df, "test", target_column),
        "train_head": train_df.head(),
        "train_info": dataframe_info_text(train_df),
        "train_describe": train_df.describe(include="all").transpose(),
        "target_distribution": pd.to_numeric(train_df[target_column], errors="coerce").describe().to_frame("price"),
        "correlation_matrix": correlation_matrix,
        "top_price_correlations": top_price_correlations,
        "split_table": pd.DataFrame(
            {
                "split": ["raw", "train", "validation", "test"],
                "rows": [len(raw_df), len(train_df), len(val_df), len(test_df)],
                "columns": [raw_df.shape[1], train_df.shape[1], val_df.shape[1], test_df.shape[1]],
                "missing_values": [
                    int(sum(missing_mask(raw_df[column]).sum() for column in raw_df.columns)),
                    int(sum(missing_mask(train_df[column]).sum() for column in train_df.columns)),
                    int(sum(missing_mask(val_df[column]).sum() for column in val_df.columns)),
                    int(sum(missing_mask(test_df[column]).sum() for column in test_df.columns)),
                ],
                "duplicate_rows": [
                    int(raw_df.duplicated().sum()),
                    int(train_df.duplicated().sum()),
                    int(val_df.duplicated().sum()),
                    int(test_df.duplicated().sum()),
                ],
            }
        ),
    }


def _save_plot(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_price_histogram(train_df: pd.DataFrame, config: ProjectConfig) -> Path:
    price = pd.to_numeric(train_df["price"], errors="coerce").dropna()
    plt.figure(figsize=(9, 5))
    plt.hist(price, bins=50, color="#2563eb", edgecolor="white")
    plt.title("Price Distribution")
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    return _save_plot(config.eda_plots_dir / "price_distribution.png")


def plot_price_boxplot(train_df: pd.DataFrame, config: ProjectConfig) -> Path:
    price = pd.to_numeric(train_df["price"], errors="coerce").dropna()
    plt.figure(figsize=(8, 4))
    plt.boxplot(price, vert=False, patch_artist=True, boxprops={"facecolor": "#93c5fd"})
    plt.title("Price Boxplot")
    plt.xlabel("Price")
    return _save_plot(config.eda_plots_dir / "price_boxplot.png")


def plot_correlation_heatmap(train_df: pd.DataFrame, config: ProjectConfig) -> Path:
    numeric_df = train_df.apply(pd.to_numeric, errors="coerce").select_dtypes(include=[np.number])
    correlation = numeric_df.corr(numeric_only=True)
    plt.figure(figsize=(12, 9))
    image = plt.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(image, fraction=0.046, pad=0.04)
    plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=90)
    plt.yticks(range(len(correlation.index)), correlation.index)
    plt.title("Correlation Heatmap")
    return _save_plot(config.eda_plots_dir / "correlation_heatmap.png")


def plot_top_correlation_heatmap(train_df: pd.DataFrame, config: ProjectConfig, top_n: int = 10) -> Path:
    """Plot an annotated heatmap for the features most correlated with price."""

    numeric_df = train_df.apply(pd.to_numeric, errors="coerce").select_dtypes(include=[np.number])
    correlation = numeric_df.corr(numeric_only=True)
    if config.target_column not in correlation.columns:
        raise KeyError(f"{config.target_column} column is required for top correlation heatmap")
    top_features = (
        correlation[config.target_column]
        .drop(labels=[config.target_column], errors="ignore")
        .dropna()
        .abs()
        .sort_values(ascending=False)
        .head(top_n)
        .index
        .tolist()
    )
    selected_columns = [config.target_column, *top_features]
    selected_correlation = correlation.loc[selected_columns, selected_columns]
    plt.figure(figsize=(10, 8))
    image = plt.imshow(selected_correlation, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(image, fraction=0.046, pad=0.04)
    plt.xticks(range(len(selected_columns)), selected_columns, rotation=45, ha="right")
    plt.yticks(range(len(selected_columns)), selected_columns)
    for row_index in range(len(selected_columns)):
        for column_index in range(len(selected_columns)):
            value = selected_correlation.iloc[row_index, column_index]
            text_color = "white" if abs(value) >= 0.55 else "#111827"
            plt.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
            )
    plt.title("Top Correlation Heatmap with Values")
    return _save_plot(config.eda_plots_dir / "top_correlation_heatmap_annotated.png")


def plot_sqft_living_vs_price(train_df: pd.DataFrame, config: ProjectConfig) -> Path:
    x = pd.to_numeric(train_df["sqft_living"], errors="coerce")
    y = pd.to_numeric(train_df["price"], errors="coerce")
    mask = x.notna() & y.notna()
    plt.figure(figsize=(8, 5))
    plt.scatter(x[mask], y[mask], alpha=0.35, s=12, color="#0891b2")
    plt.title("Sqft Living vs Price")
    plt.xlabel("sqft_living")
    plt.ylabel("price")
    return _save_plot(config.eda_plots_dir / "sqft_living_vs_price.png")


def plot_feature_histograms(train_df: pd.DataFrame, config: ProjectConfig) -> Path:
    """Plot histograms for important numeric features."""

    candidate_columns = ["price", "sqft_living", "bedrooms", "bathrooms", "grade", "sqft_lot"]
    columns = [column for column in candidate_columns if column in train_df.columns]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.flatten()
    for axis, column in zip(axes, columns):
        values = pd.to_numeric(train_df[column], errors="coerce").dropna()
        axis.hist(values, bins=35, color="#4f46e5", edgecolor="white")
        axis.set_title(column)
        axis.set_xlabel(column)
        axis.set_ylabel("Frequency")
    for axis in axes[len(columns) :]:
        axis.axis("off")
    fig.suptitle("Histograms of Important Features", fontsize=14)
    return _save_plot(config.eda_plots_dir / "important_feature_histograms.png")


def plot_price_by_zipcode(train_df: pd.DataFrame, config: ProjectConfig) -> Path:
    """Plot median price for frequent zipcodes to inspect location effect."""

    if "zipcode" not in train_df.columns:
        raise KeyError("zipcode column is required for price by location visualization")
    plot_df = train_df[["zipcode", "price"]].copy()
    plot_df["zipcode"] = plot_df["zipcode"].astype(str)
    plot_df["price"] = pd.to_numeric(plot_df["price"], errors="coerce")
    top_zipcodes = plot_df["zipcode"].value_counts().head(25).index
    median_price = (
        plot_df[plot_df["zipcode"].isin(top_zipcodes)]
        .dropna(subset=["price"])
        .groupby("zipcode")["price"]
        .median()
        .sort_values(ascending=False)
    )
    plt.figure(figsize=(12, 6))
    plt.bar(median_price.index, median_price.values, color="#0f766e")
    plt.title("Median Price by Zipcode")
    plt.xlabel("Zipcode")
    plt.ylabel("Median Price")
    plt.xticks(rotation=60, ha="right")
    return _save_plot(config.eda_plots_dir / "price_by_zipcode.png")


def plot_important_features_pairplot(train_df: pd.DataFrame, config: ProjectConfig) -> Path:
    """Plot a scatter-matrix pairplot for key numeric features without seaborn."""

    candidate_columns = ["price", "sqft_living", "grade", "bathrooms", "bedrooms"]
    columns = [column for column in candidate_columns if column in train_df.columns]
    plot_df = train_df[columns].apply(pd.to_numeric, errors="coerce").dropna()
    if len(plot_df) > 1000:
        plot_df = plot_df.sample(n=1000, random_state=config.random_state)
    axes = scatter_matrix(
        plot_df,
        figsize=(11, 11),
        diagonal="hist",
        alpha=0.35,
        color="#2563eb",
        hist_kwds={"bins": 30, "edgecolor": "white"},
    )
    for axis_row in axes:
        for axis in axis_row:
            axis.xaxis.label.set_rotation(45)
            axis.yaxis.label.set_rotation(0)
            axis.yaxis.labelpad = 40
    plt.suptitle("Pairplot of Important Features", y=1.02, fontsize=14)
    return _save_plot(config.eda_plots_dir / "important_features_pairplot.png")


def run_phase_3_data_overview_eda(
    raw_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: ProjectConfig,
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Create data overview summaries and required EDA plots from training data."""

    overview_summary = create_data_overview_summary(raw_df, train_df, val_df, test_df, config.target_column)
    eda_plot_paths = {
        "price_distribution": plot_price_histogram(train_df, config),
        "price_boxplot": plot_price_boxplot(train_df, config),
        "correlation_heatmap": plot_correlation_heatmap(train_df, config),
        "top_correlation_heatmap_annotated": plot_top_correlation_heatmap(train_df, config),
        "sqft_living_vs_price": plot_sqft_living_vs_price(train_df, config),
        "important_feature_histograms": plot_feature_histograms(train_df, config),
        "price_by_zipcode": plot_price_by_zipcode(train_df, config),
        "important_features_pairplot": plot_important_features_pairplot(train_df, config),
    }
    return overview_summary, eda_plot_paths
