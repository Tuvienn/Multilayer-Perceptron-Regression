"""Phase 6: split processed dataframes into X features and y target."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .table_display import style_colored_table as shared_style_colored_table


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

def drop_leakage_features(
    X: pd.DataFrame,
    leakage_features: list[str] | None = None,
) -> pd.DataFrame:
    """Remove features that were created from the target value."""

    active_leakage_features = leakage_features or LEAKAGE_FEATURES
    columns_to_drop = [column for column in active_leakage_features if column in X.columns]
    if columns_to_drop:
        X = X.drop(columns=columns_to_drop)
    return X


def split_features_target(
    df: pd.DataFrame,
    target_column: str = "price",
    log_transform_target: bool = True,
    leakage_features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split one dataframe into feature matrix and target vector."""

    if target_column not in df.columns:
        raise KeyError(f"Target column not found: {target_column}")
    X = df.drop(columns=[target_column]).copy()
    X = drop_leakage_features(X, leakage_features)
    y = pd.to_numeric(df[target_column], errors="coerce")
    valid_target = y.notna()
    X = X.loc[valid_target].reset_index(drop=True)
    y = y.loc[valid_target].reset_index(drop=True)
    if log_transform_target:
        y = np.log1p(y)
    return X, pd.Series(y, name=target_column)


def inverse_log_transform(values) -> np.ndarray:
    return np.expm1(np.asarray(values, dtype=float))


def get_inverse_target_transform(log_transform_target: bool) -> Callable | None:
    return inverse_log_transform if log_transform_target else None


def build_xy_shape_table(
    original_splits: dict[str, pd.DataFrame],
    X_splits: dict[str, pd.DataFrame],
    y_splits: dict[str, pd.Series],
    target_column: str,
    log_transform_target: bool,
) -> pd.DataFrame:
    """Summarize Phase 6 input/output shapes by split."""

    rows = []
    for split_name, original_df in original_splits.items():
        X = X_splits[split_name]
        y = y_splits[split_name]
        rows.append(
            {
                "split": split_name,
                "source_rows": len(original_df),
                "source_columns": original_df.shape[1],
                "X_rows": X.shape[0],
                "X_features": X.shape[1],
                "y_rows": len(y),
                "target_column": target_column,
                "target_transform": f"log1p({target_column})" if log_transform_target else target_column,
            }
        )
    return pd.DataFrame(rows)


def build_leakage_check_table(
    original_splits: dict[str, pd.DataFrame],
    X_splits: dict[str, pd.DataFrame],
    leakage_features: list[str] | None = None,
) -> pd.DataFrame:
    """Show whether leakage-risk columns were removed from X."""

    active_leakage_features = leakage_features or LEAKAGE_FEATURES
    rows = []
    for split_name, original_df in original_splits.items():
        X = X_splits[split_name]
        for feature in active_leakage_features:
            rows.append(
                {
                    "split": split_name,
                    "feature": feature,
                    "exists_before_split": feature in original_df.columns,
                    "exists_in_X": feature in X.columns,
                    "status": "Removed" if feature in original_df.columns and feature not in X.columns else "Not present",
                    "reason": "Created from target price",
                }
            )
    return pd.DataFrame(rows)


def build_target_distribution_table(
    original_splits: dict[str, pd.DataFrame],
    y_splits: dict[str, pd.Series],
    target_column: str,
    log_transform_target: bool,
) -> pd.DataFrame:
    """Compare original target distribution with the y used for modeling."""

    rows = []
    for split_name, original_df in original_splits.items():
        original_y = pd.to_numeric(original_df[target_column], errors="coerce").dropna()
        model_y = pd.to_numeric(y_splits[split_name], errors="coerce").dropna()
        rows.append(
            {
                "split": split_name,
                "original_min": original_y.min(),
                "original_median": original_y.median(),
                "original_max": original_y.max(),
                "original_skew": original_y.skew(),
                "model_y_min": model_y.min(),
                "model_y_median": model_y.median(),
                "model_y_max": model_y.max(),
                "model_y_skew": model_y.skew(),
                "model_target": f"log1p({target_column})" if log_transform_target else target_column,
            }
        )
    return pd.DataFrame(rows)


def build_target_transform_table(target_column: str, log_transform_target: bool) -> pd.DataFrame:
    """Explain the target transform and inverse transform."""

    if log_transform_target:
        transform = f"y = log1p({target_column})"
        inverse_transform = "price_pred = expm1(y_pred)"
        reason = "Reduce right skew and make training more stable"
    else:
        transform = f"y = {target_column}"
        inverse_transform = "No inverse transform needed"
        reason = "Use original target scale"
    return pd.DataFrame(
        [
            {
                "item": "Target split",
                "value": f"X drops {target_column}; y keeps {target_column}",
                "reason": "Separate model inputs from label",
            },
            {
                "item": "Transform",
                "value": transform,
                "reason": reason,
            },
            {
                "item": "Inverse transform",
                "value": inverse_transform,
                "reason": "Return predictions to house-price scale",
            },
        ]
    )


def build_phase_6_summary(
    train_feature_df: pd.DataFrame,
    val_feature_df: pd.DataFrame,
    test_feature_df: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target_column: str = "price",
    log_transform_target: bool = True,
    leakage_features: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Build colored-table friendly summaries for Phase 6."""

    original_splits = {
        "train": train_feature_df,
        "validation": val_feature_df,
        "test": test_feature_df,
    }
    X_splits = {
        "train": X_train,
        "validation": X_val,
        "test": X_test,
    }
    y_splits = {
        "train": y_train,
        "validation": y_val,
        "test": y_test,
    }
    return {
        "shape_table": build_xy_shape_table(
            original_splits,
            X_splits,
            y_splits,
            target_column,
            log_transform_target,
        ),
        "leakage_check": build_leakage_check_table(original_splits, X_splits, leakage_features),
        "target_distribution": build_target_distribution_table(
            original_splits,
            y_splits,
            target_column,
            log_transform_target,
        ),
        "target_transform": build_target_transform_table(target_column, log_transform_target),
    }


def display_phase_6_summary(summary: dict[str, pd.DataFrame]) -> None:
    """Display Phase 6 summary tables in a notebook, with a plain fallback."""

    sections = [
        ("### X/y split shape check", "shape_table", "Phase 6 X/y outputs"),
        ("### Leakage check", "leakage_check", "Leakage-risk features removed from X"),
        ("### Target distribution", "target_distribution", "Target distribution before/after transform"),
        ("### Target transform", "target_transform", "Target transform and inverse transform"),
    ]
    try:
        from IPython.display import Markdown, display
    except ModuleNotFoundError:
        for title, key, _caption in sections:
            print(title.replace("#", "").strip())
            print(summary[key])
        return

    for title, key, caption in sections:
        display(Markdown(title))
        display(style_colored_table(summary[key], caption))


def run_phase_6_split_data_xy(
    train_feature_df: pd.DataFrame,
    val_feature_df: pd.DataFrame,
    test_feature_df: pd.DataFrame,
    target_column: str = "price",
    log_transform_target: bool = True,
    leakage_features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, Callable | None]:
    """Split train/validation/test dataframes into X and y."""

    X_train, y_train = split_features_target(train_feature_df, target_column, log_transform_target, leakage_features)
    X_val, y_val = split_features_target(val_feature_df, target_column, log_transform_target, leakage_features)
    X_test, y_test = split_features_target(test_feature_df, target_column, log_transform_target, leakage_features)
    return X_train, y_train, X_val, y_val, X_test, y_test, get_inverse_target_transform(log_transform_target)
