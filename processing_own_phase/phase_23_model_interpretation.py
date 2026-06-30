"""Phase 23: interpret model behavior with feature and residual analysis."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp") / "matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .phase_1_import_library import ProjectConfig
from .table_display import style_colored_table as shared_style_colored_table


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


def _save_plot(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def _ensure_prediction_errors(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Return prediction table with residual and error columns."""

    required_columns = ["actual_price", "predicted_price"]
    missing_columns = [column for column in required_columns if column not in predictions_df.columns]
    if missing_columns:
        raise ValueError(f"predictions_df is missing required columns: {missing_columns}")
    output_df = predictions_df.copy()
    output_df["residual"] = output_df["actual_price"] - output_df["predicted_price"]
    output_df["absolute_error"] = output_df["residual"].abs()
    output_df["absolute_percentage_error"] = np.where(
        output_df["actual_price"] != 0,
        output_df["absolute_error"] / output_df["actual_price"] * 100,
        np.nan,
    )
    return output_df


def save_random_forest_feature_importance(
    random_forest_model,
    feature_names: list[str],
    config: ProjectConfig,
    top_n: int = 25,
) -> tuple[pd.DataFrame, Path]:
    """Save Random Forest feature importance as an interpretation aid."""

    if not hasattr(random_forest_model, "feature_importances_"):
        raise AttributeError("Provided model does not have feature_importances_.")
    importances = random_forest_model.feature_importances_
    if len(feature_names) != len(importances):
        raise ValueError(
            f"feature_names length ({len(feature_names)}) does not match "
            f"feature_importances_ length ({len(importances)})."
        )

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)

    output_path = config.results_dir / "random_forest_feature_importance.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(output_path, index=False)

    top_features = importance_df.head(top_n).sort_values("importance")
    plot_path = config.comparison_plots_dir / "random_forest_feature_importance.png"
    plt.figure(figsize=(10, 8))
    plt.barh(top_features["feature"], top_features["importance"], color="#2563eb")
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.grid(True, axis="x", color="#cbd5e1", alpha=0.45)
    return importance_df, _save_plot(plot_path)


def build_residual_summary(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Build residual summary from MLP predictions."""

    working_df = _ensure_prediction_errors(predictions_df)
    residuals = working_df["residual"]
    return pd.DataFrame(
        [
            {
                "mean_residual": float(residuals.mean()),
                "median_residual": float(residuals.median()),
                "std_residual": float(residuals.std()),
                "mean_absolute_error": float(working_df["absolute_error"].mean()),
                "median_absolute_error": float(working_df["absolute_error"].median()),
                "max_absolute_error": float(working_df["absolute_error"].max()),
                "mean_absolute_percentage_error": float(working_df["absolute_percentage_error"].mean()),
            }
        ]
    )


def save_residual_summary(predictions_df: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    """Save residual summary to output/results."""

    summary_df = build_residual_summary(predictions_df)
    summary_path = config.results_dir / "residual_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    return summary_df


def build_error_by_price_segment(predictions_df: pd.DataFrame, bins: int = 5) -> pd.DataFrame:
    """Summarize prediction error by actual price segment."""

    working_df = _ensure_prediction_errors(predictions_df)
    working_df["price_segment"] = pd.qcut(
        working_df["actual_price"],
        q=bins,
        duplicates="drop",
    )
    segment_df = (
        working_df.groupby("price_segment", observed=True)
        .agg(
            samples=("actual_price", "size"),
            mean_actual_price=("actual_price", "mean"),
            mean_predicted_price=("predicted_price", "mean"),
            mean_residual=("residual", "mean"),
            mean_absolute_error=("absolute_error", "mean"),
            mean_absolute_percentage_error=("absolute_percentage_error", "mean"),
        )
        .reset_index()
    )
    segment_df["price_segment"] = segment_df["price_segment"].astype(str)
    return segment_df


def save_error_by_price_segment(predictions_df: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    """Save error-by-price-segment table."""

    segment_df = build_error_by_price_segment(predictions_df)
    output_path = config.results_dir / "error_by_price_segment.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    segment_df.to_csv(output_path, index=False)
    return segment_df


def build_interpretation_notes() -> pd.DataFrame:
    """Explain how to interpret Phase 23 outputs."""

    return pd.DataFrame(
        [
            {
                "topic": "Why Random Forest importance",
                "note": "MLP is harder to interpret directly, so Random Forest gives a useful feature-importance reference.",
            },
            {
                "topic": "Residual mean",
                "note": "Positive mean residual means predictions are lower than actual prices on average.",
            },
            {
                "topic": "Error by segment",
                "note": "Higher error in expensive segments means the model struggles more with high-value homes.",
            },
            {
                "topic": "Important caution",
                "note": "Feature importance from Random Forest explains the baseline model, not the exact inner logic of the MLP.",
            },
        ]
    )


def build_phase_23_summary(
    rf_importance_df: pd.DataFrame,
    residual_summary_df: pd.DataFrame,
    error_by_segment_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Build display-ready Phase 23 summary tables."""

    return {
        "top_feature_importance": rf_importance_df.head(20).reset_index(drop=True),
        "residual_summary": residual_summary_df,
        "error_by_price_segment": error_by_segment_df,
        "interpretation_notes": build_interpretation_notes(),
    }


def display_phase_23_summary(summary: dict[str, pd.DataFrame]) -> None:
    """Display Phase 23 summary tables in a notebook, with a plain fallback."""

    sections = [
        ("### Top Random Forest feature importance", "top_feature_importance", "Top feature importance reference"),
        ("### Residual summary", "residual_summary", "MLP residual summary"),
        ("### Error by price segment", "error_by_price_segment", "MLP error by actual price segment"),
        ("### Interpretation notes", "interpretation_notes", "How to read Phase 23 outputs"),
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
        gradient_columns = [
            column
            for column in ["importance", "mean_absolute_error", "mean_absolute_percentage_error"]
            if column in summary[key].columns
        ]
        display(style_colored_table(summary[key], caption, gradient_columns=gradient_columns or None))


def run_phase_23_model_interpretation(
    fitted_baseline_models: dict[str, object],
    feature_names: list[str],
    predictions_df: pd.DataFrame,
    config: ProjectConfig,
    random_forest_key: str = "Random Forest Regression",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], dict[str, Path]]:
    """Run Phase 23 interpretation helpers."""

    if random_forest_key not in fitted_baseline_models:
        raise KeyError(f"{random_forest_key!r} not found in fitted_baseline_models.")
    random_forest_model = fitted_baseline_models[random_forest_key]

    rf_importance_df, rf_importance_plot = save_random_forest_feature_importance(
        random_forest_model=random_forest_model,
        feature_names=feature_names,
        config=config,
    )
    residual_summary_df = save_residual_summary(predictions_df, config)
    error_by_segment_df = save_error_by_price_segment(predictions_df, config)
    summary = build_phase_23_summary(rf_importance_df, residual_summary_df, error_by_segment_df)
    output_paths = {
        "random_forest_feature_importance_csv": config.results_dir / "random_forest_feature_importance.csv",
        "random_forest_feature_importance_plot": rf_importance_plot,
        "residual_summary": config.results_dir / "residual_summary.csv",
        "error_by_price_segment": config.results_dir / "error_by_price_segment.csv",
    }
    return rf_importance_df, residual_summary_df, summary, output_paths
