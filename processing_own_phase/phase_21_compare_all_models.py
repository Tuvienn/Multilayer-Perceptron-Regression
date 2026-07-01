"""Phase 21: compare the MLP model with scikit-learn baselines."""

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


CORE_METRIC_COLUMNS = ["MAE", "MSE", "RMSE", "R2"]
TIME_COLUMNS = [
    "training_time_seconds",
    "inference_time_seconds",
    "inference_time_mean_seconds",
    "inference_time_std_seconds",
]
COUNT_COLUMNS = ["inference_time_repeats"]
METRIC_LABELS = {
    "training_time_seconds": "Training time (seconds)",
    "inference_time_seconds": "Mean inference time (seconds)",
    "inference_time_mean_seconds": "Mean inference time (seconds)",
    "inference_time_std_seconds": "Inference time std (seconds)",
}


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


def validate_model_results(df: pd.DataFrame, name: str) -> None:
    """Validate a model metrics table."""

    if df is None or df.empty:
        raise ValueError(f"{name} is empty. Run the previous model evaluation phase first.")
    required_columns = ["model", *CORE_METRIC_COLUMNS]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{name} is missing required columns: {missing_columns}")


def normalize_model_result_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure comparison tables have the same columns and numeric dtypes."""

    output_df = df.copy()
    if "inference_time_mean_seconds" not in output_df.columns and "inference_time_seconds" in output_df.columns:
        output_df["inference_time_mean_seconds"] = output_df["inference_time_seconds"]
    if "inference_time_seconds" not in output_df.columns and "inference_time_mean_seconds" in output_df.columns:
        output_df["inference_time_seconds"] = output_df["inference_time_mean_seconds"]
    for column in TIME_COLUMNS:
        if column not in output_df.columns:
            output_df[column] = pd.NA
    for column in COUNT_COLUMNS:
        if column not in output_df.columns:
            output_df[column] = pd.NA
    if "target_scale" not in output_df.columns:
        output_df["target_scale"] = "not specified"

    for column in CORE_METRIC_COLUMNS + TIME_COLUMNS + COUNT_COLUMNS:
        if column in output_df.columns:
            output_df[column] = pd.to_numeric(output_df[column], errors="coerce")
    return output_df


def combine_model_results(mlp_metrics_df: pd.DataFrame, comparison_df: pd.DataFrame) -> pd.DataFrame:
    """Combine MLP and baseline model metric tables."""

    validate_model_results(mlp_metrics_df, "mlp_metrics_df")
    validate_model_results(comparison_df, "comparison_df")
    mlp_df = normalize_model_result_columns(mlp_metrics_df)
    baseline_df = normalize_model_result_columns(comparison_df)
    combined_df = pd.concat([baseline_df, mlp_df], ignore_index=True)
    ordered_columns = [
        "model",
        "MAE",
        "MSE",
        "RMSE",
        "R2",
        "training_time_seconds",
        "inference_time_seconds",
        "inference_time_mean_seconds",
        "inference_time_std_seconds",
        "inference_time_repeats",
        "target_scale",
    ]
    available_columns = [column for column in ordered_columns if column in combined_df.columns]
    return combined_df[available_columns].copy()


def add_metric_ranks(model_comparison_df: pd.DataFrame) -> pd.DataFrame:
    """Add metric ranks so the best model is easy to identify."""

    ranked_df = model_comparison_df.copy()
    ranked_df["mae_rank"] = ranked_df["MAE"].rank(method="min", ascending=True)
    ranked_df["rmse_rank"] = ranked_df["RMSE"].rank(method="min", ascending=True)
    ranked_df["r2_rank"] = ranked_df["R2"].rank(method="min", ascending=False)
    if ranked_df["training_time_seconds"].notna().any():
        ranked_df["training_time_rank"] = ranked_df["training_time_seconds"].rank(method="min", ascending=True)
    else:
        ranked_df["training_time_rank"] = pd.NA
    if ranked_df["inference_time_seconds"].notna().any():
        ranked_df["inference_time_rank"] = ranked_df["inference_time_seconds"].rank(method="min", ascending=True)
    else:
        ranked_df["inference_time_rank"] = pd.NA

    score_columns = ["mae_rank", "rmse_rank", "r2_rank"]
    ranked_df["accuracy_rank_score"] = ranked_df[score_columns].sum(axis=1)
    ranked_df = ranked_df.sort_values(["accuracy_rank_score", "RMSE"]).reset_index(drop=True)
    ranked_df.insert(0, "overall_accuracy_rank", range(1, len(ranked_df) + 1))
    rank_columns = [
        "overall_accuracy_rank",
        "mae_rank",
        "rmse_rank",
        "r2_rank",
        "training_time_rank",
        "inference_time_rank",
        "accuracy_rank_score",
    ]
    ranked_df[rank_columns] = ranked_df[rank_columns].round(0)
    return ranked_df


def build_best_model_table(model_comparison_df: pd.DataFrame) -> pd.DataFrame:
    """Build a table of the best model by each metric."""

    metric_specs = [
        ("MAE", True, "Lower is better; average absolute error"),
        ("MSE", True, "Lower is better; squared error penalizes large mistakes"),
        ("RMSE", True, "Lower is better; same unit as price"),
        ("R2", False, "Higher is better; explained target variation"),
        ("training_time_seconds", True, "Lower is faster to train"),
        ("inference_time_seconds", True, "Lower mean time is faster to predict"),
    ]
    rows = []
    for metric, lower_is_better, meaning in metric_specs:
        if metric not in model_comparison_df.columns:
            continue
        metric_df = model_comparison_df.dropna(subset=[metric])
        if metric_df.empty:
            continue
        best_row = metric_df.sort_values(metric, ascending=lower_is_better).iloc[0]
        rows.append(
            {
                "metric": metric,
                "best_model": best_row["model"],
                "best_value": float(best_row[metric]),
                "preference": "lower" if lower_is_better else "higher",
                "meaning": meaning,
            }
        )
    return pd.DataFrame(rows)


def build_comparison_rule_table() -> pd.DataFrame:
    """Explain why Phase 21 comparison is fair."""

    return pd.DataFrame(
        [
            {
                "rule": "Same test set",
                "phase_21_check": "MLP and baselines are evaluated on the same testing split",
            },
            {
                "rule": "Same preprocessing",
                "phase_21_check": "All models use processed features from the same Phase 7 pipeline",
            },
            {
                "rule": "Same metrics",
                "phase_21_check": "MAE, MSE, RMSE and R2 are compared for every model",
            },
            {
                "rule": "Same reporting scale",
                "phase_21_check": "Metrics should be on original price scale after inverse transform",
            },
            {
                "rule": "Accuracy and cost",
                "phase_21_check": "Training time and mean inference time are kept beside prediction metrics",
            },
        ]
    )


def plot_metric_bar(
    model_comparison_df: pd.DataFrame,
    metric: str,
    config: ProjectConfig,
    lower_is_better: bool = True,
) -> Path:
    """Save a bar chart for one comparison metric."""

    metric_df = model_comparison_df.dropna(subset=[metric]).copy()
    if metric_df.empty:
        raise ValueError(f"No non-missing values available for metric: {metric}")
    sorted_df = metric_df.sort_values(metric, ascending=lower_is_better)
    colors = np.where(sorted_df["model"].str.contains("MLP", case=False, na=False), "#2563eb", "#0f766e")
    metric_label = METRIC_LABELS.get(metric, metric)
    plt.figure(figsize=(9, 5.5))
    plt.bar(sorted_df["model"], sorted_df[metric], color=colors)
    plt.title(f"Model Comparison by {metric_label}")
    plt.xlabel("Model")
    plt.ylabel(metric_label)
    plt.xticks(rotation=20, ha="right")
    plt.grid(True, axis="y", color="#cbd5e1", alpha=0.45)
    return _save_plot(config.comparison_plots_dir / f"model_comparison_{metric.lower()}.png")


def create_comparison_plots(model_comparison_df: pd.DataFrame, config: ProjectConfig) -> dict[str, Path]:
    """Create Phase 21 comparison plots."""

    plots = {
        "rmse_by_model": plot_metric_bar(model_comparison_df, "RMSE", config, lower_is_better=True),
        "r2_by_model": plot_metric_bar(model_comparison_df, "R2", config, lower_is_better=False),
        "mae_by_model": plot_metric_bar(model_comparison_df, "MAE", config, lower_is_better=True),
    }
    if model_comparison_df["training_time_seconds"].notna().any():
        plots["training_time_by_model"] = plot_metric_bar(
            model_comparison_df,
            "training_time_seconds",
            config,
            lower_is_better=True,
        )
    if model_comparison_df["inference_time_seconds"].notna().any():
        plots["inference_time_by_model"] = plot_metric_bar(
            model_comparison_df,
            "inference_time_seconds",
            config,
            lower_is_better=True,
        )
    return plots


def build_phase_21_summary(model_comparison_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build display-ready Phase 21 summary tables."""

    ranked_df = add_metric_ranks(model_comparison_df)
    return {
        "model_comparison": model_comparison_df,
        "ranked_models": ranked_df,
        "best_models_by_metric": build_best_model_table(model_comparison_df),
        "comparison_rules": build_comparison_rule_table(),
    }


def display_phase_21_summary(summary: dict[str, pd.DataFrame]) -> None:
    """Display Phase 21 summary tables in a notebook, with a plain fallback."""

    sections = [
        ("### Model comparison table", "model_comparison", "All model metrics on the same test set"),
        ("### Ranked models", "ranked_models", "Model ranks by accuracy metrics"),
        ("### Best models by metric", "best_models_by_metric", "Best model for each metric"),
        ("### Fair comparison rules", "comparison_rules", "How Phase 21 keeps the comparison fair"),
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
        gradient_columns = [column for column in ["MAE", "MSE", "RMSE", "R2"] if column in summary[key].columns]
        display(style_colored_table(summary[key], caption, gradient_columns=gradient_columns or None))


def run_phase_21_compare_all_models(
    mlp_metrics_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    config: ProjectConfig,
) -> tuple[pd.DataFrame, dict[str, Path], dict[str, pd.DataFrame]]:
    """Run Phase 21: combine, rank, save and visualize all model results."""

    model_comparison_df = combine_model_results(mlp_metrics_df, comparison_df)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    model_comparison_df.to_csv(config.comparison_path, index=False)
    plots = create_comparison_plots(model_comparison_df, config)
    summary = build_phase_21_summary(model_comparison_df)
    return model_comparison_df, plots, summary