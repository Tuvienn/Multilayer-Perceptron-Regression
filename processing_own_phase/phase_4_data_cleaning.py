"""Phase 4: conservative data cleaning for each split.

Outliers are checked and reported, not removed automatically. In house-price
data, expensive homes can be valid observations caused by location, size, view,
or quality. Missing-value imputation is also left to the Phase 7 preprocessing
pipeline so it can be fit only on the training set.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from .table_display import style_colored_table as shared_style_colored_table


MISSING_TOKENS = {"", "na", "nan", "null", "none", "unknown"}
NUMERIC_COLUMNS = [
    "price",
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "grade",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "yr_renovated",
    "zipcode",
    "lat",
    "long",
    "sqft_living15",
    "sqft_lot15",
]
OUTLIER_COLUMNS = ["price", "sqft_living", "bedrooms", "bathrooms"]
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

def convert_numeric_columns(df: pd.DataFrame, columns: Iterable[str] = NUMERIC_COLUMNS) -> pd.DataFrame:
    output_df = df.replace(list(MISSING_TOKENS), np.nan).copy()
    for column in columns:
        if column in output_df.columns:
            output_df[column] = pd.to_numeric(output_df[column], errors="coerce")
    return output_df


def parse_date_column(df: pd.DataFrame) -> pd.DataFrame:
    output_df = df.copy()
    if "date" in output_df.columns:
        output_df["date"] = pd.to_datetime(output_df["date"], format="%Y%m%dT%H%M%S", errors="coerce")
    return output_df


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create basic date features during cleaning while keeping original date."""

    output_df = df.copy()
    if "date" in output_df.columns:
        output_df["sale_year"] = output_df["date"].dt.year
        output_df["sale_month"] = output_df["date"].dt.month
    return output_df


def get_invalid_value_masks(df: pd.DataFrame, target_column: str = "price") -> dict[str, pd.Series]:
    """Return invalid-value masks by column/rule."""

    masks: dict[str, pd.Series] = {}
    if target_column in df.columns:
        masks[target_column] = df[target_column].notna() & (df[target_column] <= 0)

    invalid_rules = {
        "bedrooms": lambda s: (s <= 0) | (s > 20),
        "bathrooms": lambda s: s <= 0,
        "sqft_living": lambda s: s <= 0,
        "sqft_lot": lambda s: s <= 0,
        "sqft_above": lambda s: s <= 0,
        "sqft_living15": lambda s: s <= 0,
        "sqft_lot15": lambda s: s <= 0,
        "floors": lambda s: s <= 0,
        "condition": lambda s: ~s.between(1, 5),
        "grade": lambda s: ~s.between(1, 13),
    }
    for column, rule in invalid_rules.items():
        if column in df.columns:
            masks[column] = df[column].notna() & rule(df[column])

    if {"date", "yr_built"}.issubset(df.columns):
        sale_year = df["date"].dt.year
        masks["yr_built_after_sale_year"] = df["yr_built"].notna() & sale_year.notna() & (df["yr_built"] > sale_year)

    if {"date", "yr_renovated"}.issubset(df.columns):
        sale_year = df["date"].dt.year
        masks["yr_renovated_after_sale_year"] = (
            df["yr_renovated"].notna()
            & (df["yr_renovated"] > 0)
            & sale_year.notna()
            & (df["yr_renovated"] > sale_year)
        )
    return masks


def count_invalid_values(df: pd.DataFrame, target_column: str = "price") -> int:
    return int(sum(mask.sum() for mask in get_invalid_value_masks(df, target_column).values()))


def handle_invalid_values(df: pd.DataFrame, target_column: str = "price") -> pd.DataFrame:
    output_df = df.copy()
    if target_column in output_df.columns:
        output_df = output_df[output_df[target_column].isna() | (output_df[target_column] > 0)].copy()

    invalid_rules = {
        "bedrooms": lambda s: (s <= 0) | (s > 20),
        "bathrooms": lambda s: s <= 0,
        "sqft_living": lambda s: s <= 0,
        "sqft_lot": lambda s: s <= 0,
        "sqft_above": lambda s: s <= 0,
        "sqft_living15": lambda s: s <= 0,
        "sqft_lot15": lambda s: s <= 0,
        "floors": lambda s: s <= 0,
        "condition": lambda s: ~s.between(1, 5),
        "grade": lambda s: ~s.between(1, 13),
    }
    for column, rule in invalid_rules.items():
        if column in output_df.columns:
            mask = output_df[column].notna() & rule(output_df[column])
            output_df.loc[mask, column] = np.nan

    if {"date", "yr_built"}.issubset(output_df.columns):
        sale_year = output_df["date"].dt.year
        future_built = output_df["yr_built"].notna() & sale_year.notna() & (output_df["yr_built"] > sale_year)
        output_df.loc[future_built, "yr_built"] = np.nan

    if {"date", "yr_renovated"}.issubset(output_df.columns):
        sale_year = output_df["date"].dt.year
        future_renovated = (
            output_df["yr_renovated"].notna()
            & (output_df["yr_renovated"] > 0)
            & sale_year.notna()
            & (output_df["yr_renovated"] > sale_year)
        )
        output_df.loc[future_renovated, "yr_renovated"] = np.nan

    return output_df


def clean_house_data(df: pd.DataFrame, target_column: str = "price", drop_id: bool = True) -> pd.DataFrame:
    output_df = convert_numeric_columns(df)
    output_df = parse_date_column(output_df)
    output_df = add_date_features(output_df)
    output_df = output_df.drop_duplicates().copy()
    output_df = handle_invalid_values(output_df, target_column=target_column)
    if drop_id and "id" in output_df.columns:
        output_df = output_df.drop(columns=["id"])
    return output_df.reset_index(drop=True)


def prepare_for_audit(df: pd.DataFrame) -> pd.DataFrame:
    """Apply non-destructive type normalization for audit counts."""

    return add_date_features(parse_date_column(convert_numeric_columns(df)))


def count_missing_values(df: pd.DataFrame) -> int:
    return int(df.isna().sum().sum())


def count_outlier_flags(df: pd.DataFrame, column: str) -> int:
    if column not in df.columns:
        return 0
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        return 0
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return int(((values < lower_bound) | (values > upper_bound)).sum())


def build_outlier_table(split_frames: dict[str, tuple[pd.DataFrame, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for split, (before_df, after_df) in split_frames.items():
        before_audit_df = prepare_for_audit(before_df)
        after_audit_df = prepare_for_audit(after_df)
        for column in OUTLIER_COLUMNS:
            rows.append(
                {
                    "split": split,
                    "feature": column,
                    "outlier_flags_before": count_outlier_flags(before_audit_df, column),
                    "outlier_flags_after": count_outlier_flags(after_audit_df, column),
                    "action": "Flag only, not removed automatically",
                }
            )
    return pd.DataFrame(rows)


def summarize_cleaning_split(
    split: str,
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    target_column: str = "price",
) -> dict[str, Any]:
    before_audit_df = prepare_for_audit(before_df)
    after_audit_df = prepare_for_audit(after_df)
    columns_removed = sorted(set(before_audit_df.columns) - set(after_audit_df.columns))
    return {
        "split": split,
        "rows_before": len(before_df),
        "rows_after": len(after_df),
        "rows_removed": len(before_df) - len(after_df),
        "columns_before": before_audit_df.shape[1],
        "columns_after": after_df.shape[1],
        "columns_removed": ", ".join(columns_removed) if columns_removed else "-",
        "missing_before": count_missing_values(before_audit_df),
        "missing_after": count_missing_values(after_audit_df),
        "duplicate_rows_before": int(before_audit_df.duplicated().sum()),
        "duplicate_rows_after": int(after_audit_df.duplicated().sum()),
        "invalid_values_before": count_invalid_values(before_audit_df, target_column),
        "invalid_values_after": count_invalid_values(after_audit_df, target_column),
        "date_processed": "date" in before_audit_df.columns,
        "sale_year_created": "sale_year" in after_df.columns,
        "sale_month_created": "sale_month" in after_df.columns,
    }


def build_cleaning_steps_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "step": "Missing values",
                "phase_4_action": "Standardize missing tokens to NaN; imputation is deferred to Phase 7",
                "reason": "Imputer must be fit only on training data to prevent leakage",
            },
            {
                "step": "Duplicates",
                "phase_4_action": "Drop duplicate rows inside each split",
                "reason": "Duplicate records can bias model learning and evaluation",
            },
            {
                "step": "Outliers",
                "phase_4_action": "Flag IQR outliers for price, sqft_living, bedrooms, bathrooms",
                "reason": "House-price outliers can be valid luxury/location-driven homes",
            },
            {
                "step": "Invalid values",
                "phase_4_action": "Set impossible feature values to NaN; remove non-positive target rows",
                "reason": "Later preprocessing handles NaN consistently",
            },
            {
                "step": "Remove unnecessary columns",
                "phase_4_action": "Drop id column",
                "reason": "Identifier has no predictive meaning for regression",
            },
            {
                "step": "Date processing",
                "phase_4_action": "Parse date and create sale_year, sale_month",
                "reason": "Date parts are useful and easier for later feature engineering",
            },
        ]
    )


def build_cleaning_summary(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_clean_df: pd.DataFrame,
    val_clean_df: pd.DataFrame,
    test_clean_df: pd.DataFrame,
    target_column: str = "price",
) -> dict[str, pd.DataFrame]:
    split_frames = {
        "train": (train_df, train_clean_df),
        "validation": (val_df, val_clean_df),
        "test": (test_df, test_clean_df),
    }
    audit_table = pd.DataFrame(
        [
            summarize_cleaning_split(split, before_df, after_df, target_column)
            for split, (before_df, after_df) in split_frames.items()
        ]
    )
    return {
        "cleaning_steps": build_cleaning_steps_table(),
        "audit_table": audit_table,
        "outlier_table": build_outlier_table(split_frames),
    }


def run_phase_4_data_cleaning(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str = "price",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Clean each split independently without fitting transformers."""

    train_clean_df = clean_house_data(train_df, target_column=target_column)
    val_clean_df = clean_house_data(val_df, target_column=target_column)
    test_clean_df = clean_house_data(test_df, target_column=target_column)
    cleaning_summary = build_cleaning_summary(
        train_df,
        val_df,
        test_df,
        train_clean_df,
        val_clean_df,
        test_clean_df,
        target_column,
    )
    return train_clean_df, val_clean_df, test_clean_df, cleaning_summary
