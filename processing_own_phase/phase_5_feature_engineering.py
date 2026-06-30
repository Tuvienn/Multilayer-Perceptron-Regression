"""Phase 5: deterministic feature engineering for house-price regression.

This phase creates domain features from cleaned data. It does not fit encoders,
scalers, imputers, PCA, or models. Those steps belong to later phases so the
pipeline can fit them on the training set only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .table_display import style_colored_table as shared_style_colored_table

from .phase_1_import_library import ProjectConfig, display_path_collection


ENGINEERED_FEATURES = [
    "house_age",
    "renovated",
    "years_since_renovation",
    "has_basement",
    "living_lot_ratio",
    "above_living_ratio",
    "basement_ratio",
    "living15_ratio",
    "lot15_ratio",
    "price_per_sqft",
]
LEAKAGE_FEATURES = ["price_per_sqft"]
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
    {"selector": "td", "props": [("border", "1px solid #e5e7eb"), ("padding", "8px")]},
    {"selector": "tbody tr:nth-child(even)", "props": [("background-color", "#f8fafc")]},
    {"selector": "tbody tr:nth-child(odd)", "props": [("background-color", "#ffffff")]},
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

def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide two numeric series and return NaN for invalid divisions."""

    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
    result = numerator / denominator
    return result.replace([np.inf, -np.inf], np.nan)


def ensure_sale_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create sale_year and sale_month if the cleaned dataframe still has date."""

    output_df = df.copy()
    if "date" in output_df.columns:
        if not pd.api.types.is_datetime64_any_dtype(output_df["date"]):
            output_df["date"] = pd.to_datetime(output_df["date"], format="%Y%m%dT%H%M%S", errors="coerce")
        if "sale_year" not in output_df.columns:
            output_df["sale_year"] = output_df["date"].dt.year
        if "sale_month" not in output_df.columns:
            output_df["sale_month"] = output_df["date"].dt.month
    return output_df


def add_house_price_features(
    df: pd.DataFrame,
    target_column: str = "price",
    drop_date: bool = True,
    include_price_per_sqft_for_eda: bool = True,
) -> pd.DataFrame:
    """Create deterministic house-price features for one split."""

    output_df = ensure_sale_date_features(df)

    if {"sale_year", "yr_built"}.issubset(output_df.columns):
        output_df["house_age"] = output_df["sale_year"] - pd.to_numeric(output_df["yr_built"], errors="coerce")
        output_df.loc[output_df["house_age"] < 0, "house_age"] = np.nan

    if "yr_renovated" in output_df.columns:
        renovation_year = pd.to_numeric(output_df["yr_renovated"], errors="coerce")
        output_df["renovated"] = (renovation_year.fillna(0) > 0).astype(int)
        if "sale_year" in output_df.columns:
            output_df["years_since_renovation"] = np.where(
                output_df["renovated"].eq(1),
                pd.to_numeric(output_df["sale_year"], errors="coerce") - renovation_year,
                0,
            )
            output_df.loc[output_df["years_since_renovation"] < 0, "years_since_renovation"] = np.nan

    if "sqft_basement" in output_df.columns:
        basement = pd.to_numeric(output_df["sqft_basement"], errors="coerce").fillna(0)
        output_df["has_basement"] = (basement > 0).astype(int)

    if {"sqft_living", "sqft_lot"}.issubset(output_df.columns):
        output_df["living_lot_ratio"] = safe_divide(output_df["sqft_living"], output_df["sqft_lot"])

    if {"sqft_above", "sqft_living"}.issubset(output_df.columns):
        output_df["above_living_ratio"] = safe_divide(output_df["sqft_above"], output_df["sqft_living"])

    if {"sqft_basement", "sqft_living"}.issubset(output_df.columns):
        output_df["basement_ratio"] = safe_divide(output_df["sqft_basement"], output_df["sqft_living"])

    if {"sqft_living", "sqft_living15"}.issubset(output_df.columns):
        output_df["living15_ratio"] = safe_divide(output_df["sqft_living"], output_df["sqft_living15"])

    if {"sqft_lot", "sqft_lot15"}.issubset(output_df.columns):
        output_df["lot15_ratio"] = safe_divide(output_df["sqft_lot"], output_df["sqft_lot15"])

    if include_price_per_sqft_for_eda and {target_column, "sqft_living"}.issubset(output_df.columns):
        output_df["price_per_sqft"] = safe_divide(output_df[target_column], output_df["sqft_living"])

    if drop_date and "date" in output_df.columns:
        output_df = output_df.drop(columns=["date"])
    return output_df.reset_index(drop=True)


def get_created_features(before_df: pd.DataFrame, after_df: pd.DataFrame) -> list[str]:
    return [column for column in after_df.columns if column not in before_df.columns]


def build_shape_table(
    before_splits: dict[str, pd.DataFrame],
    after_splits: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    for split_name, before_df in before_splits.items():
        after_df = after_splits[split_name]
        rows.append(
            {
                "split": split_name,
                "rows_before": len(before_df),
                "columns_before": before_df.shape[1],
                "rows_after": len(after_df),
                "columns_after": after_df.shape[1],
                "new_columns": after_df.shape[1] - before_df.shape[1],
            }
        )
    return pd.DataFrame(rows)


def build_feature_steps_table(created_features: list[str]) -> pd.DataFrame:
    feature_explanations = {
        "house_age": ("Tuoi nha tai thoi diem ban", "Yes", "sale_year - yr_built"),
        "renovated": ("Nha da tung cai tao hay chua", "Yes", "yr_renovated > 0"),
        "years_since_renovation": ("So nam tinh tu lan cai tao gan nhat", "Yes", "sale_year - yr_renovated"),
        "has_basement": ("Nha co tang ham hay khong", "Yes", "sqft_basement > 0"),
        "living_lot_ratio": ("Ty le dien tich o tren dien tich dat", "Yes", "sqft_living / sqft_lot"),
        "above_living_ratio": ("Ty le dien tich noi tren mat dat", "Yes", "sqft_above / sqft_living"),
        "basement_ratio": ("Ty le dien tich tang ham", "Yes", "sqft_basement / sqft_living"),
        "living15_ratio": ("So sanh dien tich o voi nha lan can", "Yes", "sqft_living / sqft_living15"),
        "lot15_ratio": ("So sanh dien tich dat voi nha lan can", "Yes", "sqft_lot / sqft_lot15"),
        "price_per_sqft": ("Gia moi sqft, chi dung de EDA", "No", "price / sqft_living"),
    }
    rows = []
    for feature in created_features:
        meaning, model_input, formula = feature_explanations.get(feature, ("Feature moi", "Yes", "Derived value"))
        rows.append(
            {
                "feature": feature,
                "created": "Yes",
                "model_input": model_input,
                "formula": formula,
                "meaning": meaning,
            }
        )
    return pd.DataFrame(rows)


def build_feature_quality_table(train_feature_df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for feature in features:
        if feature not in train_feature_df.columns:
            continue
        values = pd.to_numeric(train_feature_df[feature], errors="coerce")
        rows.append(
            {
                "feature": feature,
                "missing_values": int(values.isna().sum()),
                "min": values.min(),
                "median": values.median(),
                "max": values.max(),
                "skew": values.skew(),
            }
        )
    return pd.DataFrame(rows)


def build_target_skew_table(train_feature_df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    if target_column not in train_feature_df.columns:
        return pd.DataFrame(columns=["target_version", "skew", "note"])
    price = pd.to_numeric(train_feature_df[target_column], errors="coerce").dropna()
    return pd.DataFrame(
        [
            {
                "target_version": target_column,
                "skew": price.skew(),
                "note": "Dung de quan sat do lech phai cua gia nha",
            },
            {
                "target_version": f"log1p({target_column})",
                "skew": np.log1p(price).skew(),
                "note": "Se ap dung o Phase 6 neu config.log_transform_target=True",
            },
        ]
    )


def build_deferred_steps_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "item": "One-Hot Encoding cho zipcode",
                "phase": "Phase 7",
                "reason": "Can fit encoder tren train set, roi transform validation/test",
            },
            {
                "item": "log1p(price)",
                "phase": "Phase 6",
                "reason": "Day la bien doi target y, khong phai feature dau vao",
            },
            {
                "item": "PCA 2D / PCA colored by price",
                "phase": "Sau Phase 7",
                "reason": "PCA can du lieu da impute, encode va scale",
            },
            {
                "item": "Random Forest feature importance",
                "phase": "Phase 20/23",
                "reason": "Can train baseline model truoc khi lay feature importance",
            },
        ]
    )


def build_leakage_guard_table(train_feature_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in LEAKAGE_FEATURES:
        rows.append(
            {
                "feature": feature,
                "exists_in_phase_5": feature in train_feature_df.columns,
                "allowed_as_model_input": "No",
                "action": "Phase 6 drops this column from X",
                "reason": "Feature nay duoc tao tu target price nen gay data leakage",
            }
        )
    return pd.DataFrame(rows)


def build_feature_engineering_summary(
    before_splits: dict[str, pd.DataFrame],
    after_splits: dict[str, pd.DataFrame],
    target_column: str,
) -> dict[str, Any]:
    train_before_df = before_splits["train"]
    train_after_df = after_splits["train"]
    created_features = get_created_features(train_before_df, train_after_df)
    engineered_available = [feature for feature in ENGINEERED_FEATURES if feature in train_after_df.columns]

    return {
        "shape_table": build_shape_table(before_splits, after_splits),
        "feature_steps": build_feature_steps_table(created_features),
        "feature_quality": build_feature_quality_table(train_after_df, engineered_available),
        "target_skewness": build_target_skew_table(train_after_df, target_column),
        "leakage_guard": build_leakage_guard_table(train_after_df),
        "deferred_steps": build_deferred_steps_table(),
        "created_features": created_features,
        "leakage_features": LEAKAGE_FEATURES,
    }


def _save_plot(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_engineered_feature_correlation(
    train_feature_df: pd.DataFrame,
    config: ProjectConfig,
    max_features: int = 12,
) -> Path:
    """Plot an annotated correlation heatmap for engineered numeric features."""

    candidate_columns = [config.target_column] + [
        feature for feature in ENGINEERED_FEATURES if feature in train_feature_df.columns
    ]
    numeric_df = train_feature_df[candidate_columns].apply(pd.to_numeric, errors="coerce")
    correlation = numeric_df.corr(numeric_only=True).dropna(axis=0, how="all").dropna(axis=1, how="all")
    if config.target_column in correlation.columns:
        ordered_features = (
            correlation[config.target_column]
            .drop(labels=[config.target_column], errors="ignore")
            .dropna()
            .abs()
            .sort_values(ascending=False)
            .head(max_features - 1)
            .index
            .tolist()
        )
        columns = [config.target_column] + ordered_features
        correlation = correlation.loc[columns, columns]

    plt.figure(figsize=(10, 8))
    image = plt.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(image, fraction=0.046, pad=0.04)
    plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=45, ha="right")
    plt.yticks(range(len(correlation.index)), correlation.index)
    for row_index in range(len(correlation.index)):
        for col_index in range(len(correlation.columns)):
            value = correlation.iloc[row_index, col_index]
            if pd.notna(value):
                text_color = "white" if abs(value) >= 0.55 else "#111827"
                plt.text(col_index, row_index, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=8)
    plt.title("Engineered Feature Correlation Heatmap")
    return _save_plot(config.eda_plots_dir / "phase_5_engineered_feature_correlation_heatmap.png")


def create_feature_engineering_plots(train_feature_df: pd.DataFrame, config: ProjectConfig) -> list[Path]:
    """Create Phase 5 visualization outputs."""

    return [plot_engineered_feature_correlation(train_feature_df, config)]


def run_phase_5_feature_engineering(
    train_clean_df: pd.DataFrame,
    val_clean_df: pd.DataFrame,
    test_clean_df: pd.DataFrame,
    config: ProjectConfig | None = None,
    target_column: str = "price",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], list[Path]]:
    """Apply deterministic feature engineering to each split and summarize it."""

    active_target = config.target_column if config is not None else target_column
    before_splits = {
        "train": train_clean_df,
        "validation": val_clean_df,
        "test": test_clean_df,
    }
    after_splits = {
        split_name: add_house_price_features(split_df, target_column=active_target)
        for split_name, split_df in before_splits.items()
    }
    summary = build_feature_engineering_summary(before_splits, after_splits, active_target)
    plot_paths = create_feature_engineering_plots(after_splits["train"], config) if config is not None else []
    return after_splits["train"], after_splits["validation"], after_splits["test"], summary, plot_paths
