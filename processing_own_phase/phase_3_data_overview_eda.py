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
IMPORTANT_EDA_COLUMNS = ["price", "sqft_living", "sqft_lot", "bedrooms", "bathrooms", "grade"]
CORRELATION_EXCLUDED_COLUMNS = {"id", "date", "zipcode"}
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


def select_meaningful_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return numeric EDA features after removing identifiers and raw dates."""

    cols_to_drop = [c for c in df.columns if c.strip().lower() in ["id", "date", "zipcode"]]
    candidate_df = df.drop(columns=cols_to_drop)
    numeric_df = candidate_df.apply(pd.to_numeric, errors="coerce").select_dtypes(include=[np.number])
    # Drop columns that are completely empty to ensure no blank rows/cols in heatmap
    numeric_df = numeric_df.dropna(axis=1, how="all")
    meaningful_columns = [
        column
        for column in numeric_df.columns
        if numeric_df[column].notna().sum() > 1 and numeric_df[column].nunique(dropna=True) > 1
    ]
    return numeric_df[meaningful_columns]


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


def build_high_correlation_pair_table(
    correlation_matrix: pd.DataFrame,
    min_abs_correlation: float = 0.8,
    max_abs_correlation: float = 0.9,
) -> pd.DataFrame:
    """Return non-duplicate feature pairs with high absolute correlation for review."""

    rows = []
    columns = list(correlation_matrix.columns)
    for left_index, feature_1 in enumerate(columns):
        for feature_2 in columns[left_index + 1 :]:
            correlation = correlation_matrix.loc[feature_1, feature_2]
            if pd.isna(correlation):
                continue
            abs_correlation = abs(float(correlation))
            if min_abs_correlation <= abs_correlation <= max_abs_correlation:
                relationship = "duong" if correlation > 0 else "am"
                rows.append(
                    {
                        "feature_1": feature_1,
                        "feature_2": feature_2,
                        "correlation": float(correlation),
                        "note": (
                            f"Tuong quan {relationship} cao; can review kha nang trung lap thong tin "
                            "hoac multicollinearity, khong xoa feature tu dong trong EDA"
                        ),
                    }
                )
    return (
        pd.DataFrame(rows)
        .sort_values("correlation", key=lambda series: series.abs(), ascending=False)
        .reset_index(drop=True)
        if rows
        else pd.DataFrame(
            {
                "feature_1": pd.Series(dtype="str"),
                "feature_2": pd.Series(dtype="str"),
                "correlation": pd.Series(dtype="float"),
                "note": pd.Series(dtype="str"),
            }
        )
    )


def create_data_overview_summary(
    raw_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str = "price",
) -> dict[str, Any]:
    """Return split summaries for notebook display without modifying data."""

    numeric_train_df = select_meaningful_numeric_features(train_df)
    correlation_matrix = numeric_train_df.corr(numeric_only=True).dropna(axis=0, how="all").dropna(axis=1, how="all")
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
        "feature_type_table": build_feature_type_table(train_df),
        "quartile_outlier_table": build_quartile_outlier_table(train_df),
        "skewness_table": build_skewness_table(train_df),
        "correlation_matrix": correlation_matrix,
        "top_price_correlations": top_price_correlations,
        "high_correlation_pairs": build_high_correlation_pair_table(correlation_matrix),
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


def classify_feature_type(column: str, series: pd.Series) -> str:
    """Classify a dataset column as continuous, discrete, categorical, or datetime."""

    if column == "date":
        return "datetime"
    if column in {"zipcode", "waterfront", "view"}:
        return "categorical"
    numeric_series = pd.to_numeric(series, errors="coerce")
    non_missing = numeric_series.dropna()
    if non_missing.empty:
        return "unknown"
    unique_count = int(non_missing.nunique())
    unique_ratio = unique_count / len(non_missing)
    integer_like = bool(np.all(np.isclose(non_missing, np.round(non_missing))))
    if series.dtype == "object" or str(series.dtype).startswith("category"):
        return "categorical"
    if integer_like and (unique_count <= 20 or unique_ratio < 0.02):
        return "discrete"
    return "continuous"


def build_feature_type_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build a table separating continuous, discrete, and categorical fields."""

    rows = []
    for column in df.columns:
        feature_type = classify_feature_type(column, df[column])
        rows.append(
            {
                "feature": column,
                "feature_type": feature_type,
                "dtype": str(df[column].dtype),
                "unique_values": int(df[column].nunique(dropna=True)),
                "missing_values": int(missing_mask(df[column]).sum()),
                "example_values": ", ".join(df[column].dropna().astype(str).head(3).tolist()),
            }
        )
    return pd.DataFrame(rows)


def build_quartile_outlier_table(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Summarize quartiles and IQR outlier bounds for important numeric columns."""

    candidate_columns = columns or IMPORTANT_EDA_COLUMNS
    rows = []
    for column in candidate_columns:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        if values.empty:
            continue
        q1 = float(values.quantile(0.25))
        median = float(values.quantile(0.50))
        q3 = float(values.quantile(0.75))
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_count = int(((values < lower_bound) | (values > upper_bound)).sum())
        rows.append(
            {
                "feature": column,
                "min": float(values.min()),
                "q1": q1,
                "median": median,
                "q3": q3,
                "max": float(values.max()),
                "iqr": iqr,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "iqr_outlier_count": outlier_count,
                "outlier_ratio": outlier_count / len(values),
            }
        )
    return pd.DataFrame(rows)


def describe_skewness(skew_value: float) -> str:
    """Return a compact interpretation for a skewness value."""

    if pd.isna(skew_value):
        return "unknown"
    if skew_value > 1:
        return "strong right skew"
    if skew_value > 0.5:
        return "moderate right skew"
    if skew_value < -1:
        return "strong left skew"
    if skew_value < -0.5:
        return "moderate left skew"
    return "approximately symmetric"


def build_skewness_table(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Summarize skewness for important numeric columns."""

    candidate_columns = columns or IMPORTANT_EDA_COLUMNS
    rows = []
    for column in candidate_columns:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        if values.empty:
            continue
        skew_value = float(values.skew())
        rows.append(
            {
                "feature": column,
                "skew": skew_value,
                "interpretation": describe_skewness(skew_value),
                "log1p_skew": float(np.log1p(values[values >= 0]).skew()) if (values >= 0).any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


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


def plot_log_price_histogram(train_df: pd.DataFrame, config: ProjectConfig) -> Path:
    price = pd.to_numeric(train_df["price"], errors="coerce").dropna()
    log_price = np.log1p(price[price > 0])
    plt.figure(figsize=(9, 5))
    plt.hist(log_price, bins=50, color="#0f766e", edgecolor="white")
    plt.title("log1p(price) Distribution")
    plt.xlabel("log1p(price)")
    plt.ylabel("Frequency")
    return _save_plot(config.eda_plots_dir / "log_price_distribution.png")


def plot_log_price_boxplot(train_df: pd.DataFrame, config: ProjectConfig) -> Path:
    price = pd.to_numeric(train_df["price"], errors="coerce").dropna()
    log_price = np.log1p(price[price > 0])
    plt.figure(figsize=(8, 4))
    plt.boxplot(log_price, vert=False, patch_artist=True, boxprops={"facecolor": "#99f6e4"})
    plt.title("log1p(price) Boxplot")
    plt.xlabel("log1p(price)")
    return _save_plot(config.eda_plots_dir / "log_price_boxplot.png")


def plot_correlation_heatmap(train_df: pd.DataFrame, config: ProjectConfig) -> Path:
    numeric_df = select_meaningful_numeric_features(train_df)
    correlation = numeric_df.corr(numeric_only=True).dropna(axis=0, how="all").dropna(axis=1, how="all")
    plt.figure(figsize=(12, 9))
    image = plt.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(image, fraction=0.046, pad=0.04)
    plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=90)
    plt.yticks(range(len(correlation.index)), correlation.index)
    for row_index in range(len(correlation.index)):
        for column_index in range(len(correlation.columns)):
            value = correlation.iloc[row_index, column_index]
            if pd.notna(value):
                text_color = "white" if abs(value) >= 0.55 else "#111827"
                plt.text(column_index, row_index, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=5)
    plt.title("Correlation Heatmap with Values (id/date excluded)")
    return _save_plot(config.eda_plots_dir / "correlation_heatmap.png")


def plot_high_correlation_pair_scatter_plots(
    train_df: pd.DataFrame,
    high_correlation_pairs: pd.DataFrame,
    config: ProjectConfig,
    max_pairs: int = 6,
) -> Path:
    """Plot scatter charts for feature pairs that need high-correlation review."""

    output_path = config.eda_plots_dir / "high_correlation_pair_scatter_plots.png"
    if high_correlation_pairs.empty:
        plt.figure(figsize=(9, 4))
        plt.axis("off")
        plt.text(
            0.5,
            0.5,
            "No feature pairs found for 0.8 <= |correlation| <= 0.9",
            ha="center",
            va="center",
            fontsize=13,
            color="#334155",
        )
        plt.title("High Correlation Pair Analysis")
        return _save_plot(output_path)

    pairs = high_correlation_pairs.head(max_pairs)
    pair_count = len(pairs)
    column_count = 2 if pair_count > 1 else 1
    row_count = int(np.ceil(pair_count / column_count))
    fig, axes = plt.subplots(row_count, column_count, figsize=(6 * column_count, 4.5 * row_count))
    axes_array = np.atleast_1d(axes).flatten()

    for axis, (_, row) in zip(axes_array, pairs.iterrows()):
        feature_1 = str(row["feature_1"])
        feature_2 = str(row["feature_2"])
        if feature_1 not in train_df.columns or feature_2 not in train_df.columns:
            axis.axis("off")
            axis.text(0.5, 0.5, "Feature not found in train set", ha="center", va="center")
            continue

        plot_df = pd.DataFrame(
            {
                feature_1: pd.to_numeric(train_df[feature_1], errors="coerce"),
                feature_2: pd.to_numeric(train_df[feature_2], errors="coerce"),
            }
        ).dropna()
        if len(plot_df) > 2000:
            plot_df = plot_df.sample(n=2000, random_state=config.random_state)

        correlation = float(row["correlation"])
        axis.scatter(plot_df[feature_1], plot_df[feature_2], alpha=0.35, s=12, color="#2563eb")
        axis.set_title(f"{feature_1} vs {feature_2} (r={correlation:.3f})")
        axis.set_xlabel(feature_1)
        axis.set_ylabel(feature_2)
        axis.grid(alpha=0.25)

    for axis in axes_array[pair_count:]:
        axis.axis("off")

    fig.suptitle("High Correlation Pair Review (0.8 <= |r| <= 0.9)", fontsize=14)
    return _save_plot(output_path)


def estimate_kde_curve(values: pd.Series, points: int = 220, sample_size: int = 1500) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a simple Gaussian KDE curve without requiring scipy/seaborn."""

    clean_values = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if clean_values.size < 2:
        return np.array([]), np.array([])
    if clean_values.size > sample_size:
        rng = np.random.default_rng(42)
        clean_values = rng.choice(clean_values, size=sample_size, replace=False)
    value_std = float(np.std(clean_values, ddof=1))
    if value_std == 0:
        return np.array([]), np.array([])
    bandwidth = 1.06 * value_std * (clean_values.size ** (-1 / 5))
    if bandwidth <= 0:
        return np.array([]), np.array([])
    x_grid = np.linspace(float(clean_values.min()), float(clean_values.max()), points)
    density = np.zeros_like(x_grid)
    chunk_size = 300
    for start in range(0, clean_values.size, chunk_size):
        chunk = clean_values[start : start + chunk_size]
        z = (x_grid[:, None] - chunk[None, :]) / bandwidth
        density += np.exp(-0.5 * z**2).sum(axis=1)
    density = density / (clean_values.size * bandwidth * np.sqrt(2 * np.pi))
    return x_grid, density


def plot_kde_distributions(train_df: pd.DataFrame, config: ProjectConfig) -> Path:
    """Plot KDE curves for price and important numeric features."""

    columns = [column for column in IMPORTANT_EDA_COLUMNS if column in train_df.columns]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    axes = axes.flatten()
    for axis, column in zip(axes, columns):
        x_grid, density = estimate_kde_curve(train_df[column])
        if x_grid.size:
            axis.plot(x_grid, density, color="#be123c", linewidth=2)
            axis.fill_between(x_grid, density, color="#fecdd3", alpha=0.55)
        axis.set_title(f"KDE - {column}")
        axis.set_xlabel(column)
        axis.set_ylabel("Density")
        axis.grid(True, color="#cbd5e1", alpha=0.35)
    for axis in axes[len(columns) :]:
        axis.axis("off")
    fig.suptitle("KDE Plots for Important Numeric Features", fontsize=14)
    return _save_plot(config.eda_plots_dir / "important_feature_kde_plots.png")


def plot_important_feature_boxplots(train_df: pd.DataFrame, config: ProjectConfig) -> Path:
    """Plot boxplots for important numeric features."""

    columns = [column for column in IMPORTANT_EDA_COLUMNS if column in train_df.columns]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    axes = axes.flatten()
    for axis, column in zip(axes, columns):
        values = pd.to_numeric(train_df[column], errors="coerce").dropna()
        axis.boxplot(values, vert=False, patch_artist=True, boxprops={"facecolor": "#bfdbfe"})
        axis.set_title(column)
        axis.set_xlabel(column)
        axis.grid(True, axis="x", color="#cbd5e1", alpha=0.35)
    for axis in axes[len(columns) :]:
        axis.axis("off")
    fig.suptitle("Boxplots for Important Numeric Features", fontsize=14)
    return _save_plot(config.eda_plots_dir / "important_feature_boxplots.png")


def plot_top_correlation_heatmap(train_df: pd.DataFrame, config: ProjectConfig, top_n: int = 10) -> Path:
    """Plot an annotated heatmap for the features most correlated with price."""

    numeric_df = select_meaningful_numeric_features(train_df)
    correlation = numeric_df.corr(numeric_only=True).dropna(axis=0, how="all").dropna(axis=1, how="all")
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
        "log_price_distribution": plot_log_price_histogram(train_df, config),
        "log_price_boxplot": plot_log_price_boxplot(train_df, config),
        "correlation_heatmap": plot_correlation_heatmap(train_df, config),
        "high_correlation_pair_scatter_plots": plot_high_correlation_pair_scatter_plots(
            train_df,
            overview_summary["high_correlation_pairs"],
            config,
        ),
        "top_correlation_heatmap_annotated": plot_top_correlation_heatmap(train_df, config),
        "important_feature_kde_plots": plot_kde_distributions(train_df, config),
        "important_feature_boxplots": plot_important_feature_boxplots(train_df, config),
        "sqft_living_vs_price": plot_sqft_living_vs_price(train_df, config),
        "important_feature_histograms": plot_feature_histograms(train_df, config),
        "price_by_zipcode": plot_price_by_zipcode(train_df, config),
        "important_features_pairplot": plot_important_features_pairplot(train_df, config),
    }
    return overview_summary, eda_plot_paths
