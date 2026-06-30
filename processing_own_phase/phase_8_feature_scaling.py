"""Phase 8: audit feature scaling results from the Phase 7 pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .table_display import style_colored_table as shared_style_colored_table


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

def summarize_scaled_features(X_train_processed, X_val_processed, X_test_processed) -> pd.DataFrame:
    """Explain scaling output without fitting a separate scaler."""

    rows = []
    for split_name, values in [
        ("train", X_train_processed),
        ("validation", X_val_processed),
        ("test", X_test_processed),
    ]:
        array = np.asarray(values, dtype=float)
        feature_means = np.nanmean(array, axis=0)
        feature_stds = np.nanstd(array, axis=0)
        rows.append(
            {
                "split": split_name,
                "rows": array.shape[0],
                "processed_features": array.shape[1],
                "mean_of_feature_means": float(np.nanmean(feature_means)),
                "mean_abs_feature_mean": float(np.nanmean(np.abs(feature_means))),
                "mean_of_feature_stds": float(np.nanmean(feature_stds)),
                "min_value": float(np.nanmin(array)),
                "max_value": float(np.nanmax(array)),
                "nan_values": int(np.isnan(array).sum()),
                "infinite_values": int(np.isinf(array).sum()),
            }
        )
    return pd.DataFrame(rows)


def build_scaling_decision_table() -> pd.DataFrame:
    """Explain why Phase 8 audits scaling instead of fitting a scaler again."""

    return pd.DataFrame(
        [
            {
                "item": "Where StandardScaler is fitted",
                "decision": "Phase 7",
                "reason": "Scaling is part of the sklearn preprocessing pipeline",
            },
            {
                "item": "What Phase 8 does",
                "decision": "Audit only",
                "reason": "Avoid fitting another scaler and keep train-only fit logic clear",
            },
            {
                "item": "Leakage prevention",
                "decision": "No fit on validation/test",
                "reason": "Validation and test must only use transforms learned from train",
            },
            {
                "item": "One-hot columns",
                "decision": "Not centered like numeric columns",
                "reason": "Processed output contains both scaled numeric columns and one-hot categorical columns",
            },
        ]
    )


def build_phase_8_summary(X_train_processed, X_val_processed, X_test_processed) -> dict[str, pd.DataFrame]:
    """Build colored-table friendly Phase 8 summaries."""

    return {
        "scaling_summary": summarize_scaled_features(X_train_processed, X_val_processed, X_test_processed),
        "scaling_decisions": build_scaling_decision_table(),
    }


def display_phase_8_summary(summary: dict[str, pd.DataFrame]) -> None:
    """Display Phase 8 summary tables in a notebook, with a plain fallback."""

    sections = [
        ("### Scaling output check", "scaling_summary", "Processed arrays after Phase 7 scaling"),
        ("### Scaling decisions", "scaling_decisions", "Why Phase 8 does not fit another scaler"),
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
