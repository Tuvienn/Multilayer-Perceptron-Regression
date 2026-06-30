"""Shared notebook table styling helpers for all project phases."""

from __future__ import annotations

import numpy as np
import pandas as pd


TABLE_STYLES = [
    {
        "selector": "table",
        "props": [
            ("background-color", "#ffffff"),
            ("color", "#0f172a"),
            ("color-scheme", "light"),
            ("border-collapse", "collapse"),
        ],
    },
    {
        "selector": "caption",
        "props": [
            ("caption-side", "top"),
            ("font-weight", "800"),
            ("font-size", "15px"),
            ("color", "#0f172a"),
            ("background-color", "#e0f2fe"),
            ("padding", "10px 12px"),
            ("text-align", "left"),
            ("border", "1px solid #bae6fd"),
            ("border-bottom", "0"),
        ],
    },
    {
        "selector": "th",
        "props": [
            ("background-color", "#075985"),
            ("color", "#ffffff"),
            ("font-weight", "800"),
            ("border", "1px solid #bae6fd"),
            ("padding", "9px"),
            ("text-align", "left"),
        ],
    },
    {
        "selector": "td",
        "props": [
            ("border", "1px solid #dbeafe"),
            ("padding", "9px"),
            ("font-weight", "650"),
        ],
    },
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

    styled_df = df.copy()
    numeric_columns = list(styled_df.select_dtypes(include=[np.number]).columns)
    if gradient_columns is not None:
        numeric_columns = [column for column in gradient_columns if column in styled_df.columns]

    styler = (
        styled_df.style.set_caption(caption)
        .set_table_styles(TABLE_STYLES)
        .set_properties(
            **{
                "color": "#0f172a",
                "background-color": "#ffffff",
                "font-size": "13px",
                "line-height": "1.35",
            }
        )
        .format(precision=precision, na_rep="-")
    )
    if numeric_columns:
        styler = styler.background_gradient(
            cmap=cmap,
            subset=numeric_columns,
            text_color_threshold=0.55,
        )
    return styler
