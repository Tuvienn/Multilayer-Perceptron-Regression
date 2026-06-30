"""Phase 2: load raw data and split before any preprocessing.

This phase intentionally does not fill missing values, scale features,
encode categories, or create engineered features. Those steps must be fit
on the training set in later phases, then only transform validation/test.
"""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from .phase_1_import_library import ProjectConfig


def validate_split_config(config: ProjectConfig) -> None:
    """Validate split ratios before creating train/validation/test sets."""

    ratios = {
        "train_ratio": config.train_ratio,
        "validation_ratio": config.validation_ratio,
        "test_ratio": config.test_ratio,
    }
    for name, value in ratios.items():
        if value <= 0 or value >= 1:
            raise ValueError(f"{name} must be between 0 and 1, got {value}")

    total = sum(ratios.values())
    if abs(total - 1.0) > 1e-8:
        raise ValueError("train_ratio + validation_ratio + test_ratio must equal 1.0")


def resolve_data_path(config: ProjectConfig) -> Path:
    """Resolve dataset path from the project config and known data locations."""

    candidates = [
        config.data_path,
        config.root_dir / "multilayer_perceptron_regression" / "data" / "kc_house_data.csv",
        config.root_dir / "multilayer_perceptron_regression" / "data" / "KC_housing_data.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find dataset. Checked: {[str(path) for path in candidates]}")


def validate_raw_dataframe(raw_df: pd.DataFrame, config: ProjectConfig, data_path: Path) -> None:
    """Validate only raw-data requirements; do not transform the dataframe."""

    if raw_df.empty:
        raise ValueError(f"Dataset is empty: {data_path}")
    if config.target_column not in raw_df.columns:
        raise ValueError(
            f"Target column '{config.target_column}' not found in dataset columns: {list(raw_df.columns)}"
        )


def load_raw_dataframe(config: ProjectConfig) -> pd.DataFrame:
    """Load the raw house price dataset without preprocessing it."""

    data_path = resolve_data_path(config)
    raw_df = pd.read_csv(data_path)
    validate_raw_dataframe(raw_df, config, data_path)
    return raw_df


def split_raw_dataframe(
    raw_df: pd.DataFrame,
    config: ProjectConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split raw data before preprocessing to prevent data leakage."""

    validate_split_config(config)

    train_df, temp_df = train_test_split(
        raw_df,
        train_size=config.train_ratio,
        random_state=config.random_state,
        shuffle=True,
    )
    validation_fraction = config.validation_ratio / (config.validation_ratio + config.test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=validation_fraction,
        random_state=config.random_state,
        shuffle=True,
    )
    return (
        train_df.copy().reset_index(drop=True),
        val_df.copy().reset_index(drop=True),
        test_df.copy().reset_index(drop=True),
    )


def build_split_summary(
    raw_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: ProjectConfig,
) -> dict[str, Any]:
    """Create a small summary for notebook display without changing data."""

    total_rows = len(raw_df)
    return {
        "data_path": str(resolve_data_path(config)),
        "target_column": config.target_column,
        "raw_shape": raw_df.shape,
        "train_shape": train_df.shape,
        "validation_shape": val_df.shape,
        "test_shape": test_df.shape,
        "train_ratio_actual": len(train_df) / total_rows,
        "validation_ratio_actual": len(val_df) / total_rows,
        "test_ratio_actual": len(test_df) / total_rows,
        "preprocessing_done": False,
    }


def build_phase_2_html(split_summary: dict[str, Any]) -> str:
    """Build a visual HTML table for Phase 2 split results."""

    table_rows = [
        ("Raw data", split_summary["raw_shape"], 1.0, "Full dataset before split"),
        ("Train", split_summary["train_shape"], split_summary["train_ratio_actual"], "Used to fit preprocessing and model"),
        (
            "Validation",
            split_summary["validation_shape"],
            split_summary["validation_ratio_actual"],
            "Used for tuning and early stopping",
        ),
        ("Test", split_summary["test_shape"], split_summary["test_ratio_actual"], "Used only for final evaluation"),
    ]
    rows_html = "\n".join(
        f"""
        <tr>
          <td style="padding:10px 12px;border-bottom:1px solid #e5e7eb;font-weight:700;color:#111827;">{escape(name)}</td>
          <td style="padding:10px 12px;border-bottom:1px solid #e5e7eb;font-family:Menlo,Consolas,monospace;color:#334155;">{escape(str(shape))}</td>
          <td style="padding:10px 12px;border-bottom:1px solid #e5e7eb;color:#334155;">{ratio:.2%}</td>
          <td style="padding:10px 12px;border-bottom:1px solid #e5e7eb;color:#475569;">{escape(note)}</td>
        </tr>
        """
        for name, shape, ratio, note in table_rows
    )

    preprocessing_badge = "NO PREPROCESSING" if not split_summary["preprocessing_done"] else "PREPROCESSED"
    preprocessing_color = "#166534" if not split_summary["preprocessing_done"] else "#991b1b"
    preprocessing_bg = "#dcfce7" if not split_summary["preprocessing_done"] else "#fee2e2"
    return f"""
    <div style="border:1px solid #cbd5e1;border-radius:8px;overflow:hidden;background:#f8fafc;margin:8px 0 16px;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
      <div style="background:#1f2937;color:white;padding:16px 18px;">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px;flex-wrap:wrap;">
          <div>
            <div style="font-size:20px;font-weight:800;">Phase 2 - Data Collection and Early Split</div>
            <div style="font-size:13px;opacity:0.88;margin-top:4px;">Load raw CSV, then split into train / validation / test before any preprocessing.</div>
          </div>
          <span style="background:{preprocessing_bg};color:{preprocessing_color};border-radius:999px;padding:6px 10px;font-size:12px;font-weight:800;">
            {preprocessing_badge}
          </span>
        </div>
      </div>
      <div style="padding:16px 18px;">
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin-bottom:14px;">
          <div style="background:#ffffff;border:1px solid #d7dee8;border-radius:8px;padding:12px;">
            <div style="font-size:12px;color:#64748b;text-transform:uppercase;font-weight:800;">Target</div>
            <div style="font-size:22px;color:#111827;font-weight:800;margin-top:2px;">{escape(str(split_summary["target_column"]))}</div>
          </div>
          <div style="background:#ffffff;border:1px solid #d7dee8;border-radius:8px;padding:12px;">
            <div style="font-size:12px;color:#64748b;text-transform:uppercase;font-weight:800;">Total rows</div>
            <div style="font-size:22px;color:#111827;font-weight:800;margin-top:2px;">{split_summary["raw_shape"][0]:,}</div>
          </div>
          <div style="background:#ffffff;border:1px solid #d7dee8;border-radius:8px;padding:12px;">
            <div style="font-size:12px;color:#64748b;text-transform:uppercase;font-weight:800;">Total columns</div>
            <div style="font-size:22px;color:#111827;font-weight:800;margin-top:2px;">{split_summary["raw_shape"][1]:,}</div>
          </div>
        </div>

        <div style="background:#ffffff;border:1px solid #d7dee8;border-radius:8px;overflow:hidden;">
          <table style="width:100%;border-collapse:collapse;font-size:13px;">
            <thead>
              <tr style="background:#eef2f7;color:#334155;text-align:left;">
                <th style="padding:10px 12px;border-bottom:1px solid #cbd5e1;">Dataset</th>
                <th style="padding:10px 12px;border-bottom:1px solid #cbd5e1;">Shape</th>
                <th style="padding:10px 12px;border-bottom:1px solid #cbd5e1;">Actual ratio</th>
                <th style="padding:10px 12px;border-bottom:1px solid #cbd5e1;">Purpose</th>
              </tr>
            </thead>
            <tbody>
              {rows_html}
            </tbody>
          </table>
        </div>

        <div style="margin-top:12px;background:#ffffff;border:1px solid #d7dee8;border-radius:8px;padding:10px 12px;">
          <div style="font-size:12px;color:#64748b;text-transform:uppercase;font-weight:800;margin-bottom:4px;">Data source</div>
          <div style="font-family:Menlo,Consolas,monospace;font-size:12px;color:#334155;word-break:break-all;">{escape(str(split_summary["data_path"]))}</div>
        </div>
      </div>
    </div>
    """


def display_phase_2_split_summary(
    raw_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: ProjectConfig,
) -> dict[str, Any]:
    """Display Phase 2 split results as HTML in notebooks."""

    split_summary = build_split_summary(raw_df, train_df, val_df, test_df, config)
    html = build_phase_2_html(split_summary)
    try:
        from IPython.display import HTML, display
    except ModuleNotFoundError:
        print(split_summary)
        return split_summary
    display(HTML(html))
    return split_summary


def run_phase_2_collection_split(
    config: ProjectConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run Phase 2 and return raw/train/validation/test dataframes."""

    raw_df = load_raw_dataframe(config)
    train_df, val_df, test_df = split_raw_dataframe(raw_df, config)
    return raw_df, train_df, val_df, test_df
