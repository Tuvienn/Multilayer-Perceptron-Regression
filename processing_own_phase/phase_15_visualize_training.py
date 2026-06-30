"""Phase 15: visualize MLP training history from the training log."""

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


PLOT_STYLE = {
    "train": "#2563eb",
    "validation": "#dc2626",
    "single": "#0f766e",
    "accent": "#7c3aed",
    "grid": "#cbd5e1",
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


def _require_columns(training_log: pd.DataFrame, columns: list[str]) -> None:
    """Raise a helpful error if a required plot column is missing."""

    missing_columns = [column for column in columns if column not in training_log.columns]
    if missing_columns:
        raise ValueError(f"training_log is missing columns needed for Phase 15: {missing_columns}")


def _base_training_plot(title: str, ylabel: str) -> None:
    """Apply a consistent visual style to training plots."""

    plt.figure(figsize=(9, 5.2))
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True, color=PLOT_STYLE["grid"], alpha=0.45, linewidth=0.8)


def plot_two_metric_curve(
    training_log: pd.DataFrame,
    columns: tuple[str, str],
    title: str,
    ylabel: str,
    path: Path,
) -> Path:
    """Plot train and validation curves for one metric."""

    _require_columns(training_log, ["epoch", columns[0], columns[1]])
    _base_training_plot(title, ylabel)
    plt.plot(
        training_log["epoch"],
        training_log[columns[0]],
        label=columns[0],
        color=PLOT_STYLE["train"],
        linewidth=2,
        marker="o",
        markersize=3,
    )
    plt.plot(
        training_log["epoch"],
        training_log[columns[1]],
        label=columns[1],
        color=PLOT_STYLE["validation"],
        linewidth=2,
        marker="o",
        markersize=3,
    )
    plt.legend()
    return _save_plot(path)


def plot_single_metric_curve(
    training_log: pd.DataFrame,
    column: str,
    title: str,
    ylabel: str,
    path: Path,
    color: str | None = None,
) -> Path:
    """Plot a single metric curve when the column is available."""

    _require_columns(training_log, ["epoch", column])
    _base_training_plot(title, ylabel)
    plt.plot(
        training_log["epoch"],
        training_log[column],
        label=column,
        color=color or PLOT_STYLE["single"],
        linewidth=2,
        marker="o",
        markersize=3,
    )
    plt.legend()
    return _save_plot(path)


def plot_residual_distribution(y_true, y_pred, config: ProjectConfig) -> Path:
    """Optionally plot residuals after predictions are available."""

    y_true_array = np.asarray(y_true).reshape(-1)
    y_pred_array = np.asarray(y_pred).reshape(-1)
    if y_true_array.shape != y_pred_array.shape:
        raise ValueError("y_true and y_pred must have the same shape for residual plotting.")

    residuals = y_true_array - y_pred_array
    path = config.training_plots_dir / "optional_residual_distribution.png"
    plt.figure(figsize=(9, 5.2))
    plt.hist(residuals, bins=40, color=PLOT_STYLE["accent"], alpha=0.82, edgecolor="white")
    plt.axvline(0, color="#111827", linewidth=1.5, linestyle="--", label="zero error")
    plt.title("Optional Residual Distribution")
    plt.xlabel("Residual = actual - predicted")
    plt.ylabel("Frequency")
    plt.grid(True, color=PLOT_STYLE["grid"], alpha=0.35, linewidth=0.8)
    plt.legend()
    return _save_plot(path)


def build_training_plot_paths(
    training_log: pd.DataFrame,
    config: ProjectConfig,
    y_true=None,
    y_pred=None,
) -> tuple[dict[str, Path], list[dict[str, str]]]:
    """Create required training plots and skip optional plots when data is unavailable."""

    if training_log is None or training_log.empty:
        raise ValueError("training_log is empty. Run Phase 13 before Phase 15.")

    plot_paths: dict[str, Path] = {}
    skipped_plots: list[dict[str, str]] = []

    plot_paths["training_loss_curve"] = plot_two_metric_curve(
        training_log,
        ("train_loss", "validation_loss"),
        "Train Loss vs Validation Loss",
        "MSE Loss",
        config.training_plots_dir / "training_loss_curve.png",
    )
    plot_paths["mae_curve"] = plot_two_metric_curve(
        training_log,
        ("train_mae", "validation_mae"),
        "Train MAE vs Validation MAE",
        "MAE",
        config.training_plots_dir / "mae_curve.png",
    )

    if "train_rmse" in training_log.columns:
        plot_paths["validation_rmse_curve"] = plot_two_metric_curve(
            training_log,
            ("train_rmse", "validation_rmse"),
            "Train RMSE vs Validation RMSE",
            "RMSE",
            config.training_plots_dir / "validation_rmse_curve.png",
        )
    else:
        plot_paths["validation_rmse_curve"] = plot_single_metric_curve(
            training_log,
            "validation_rmse",
            "Validation RMSE Curve",
            "RMSE",
            config.training_plots_dir / "validation_rmse_curve.png",
        )

    if "train_r2" in training_log.columns:
        plot_paths["validation_r2_curve"] = plot_two_metric_curve(
            training_log,
            ("train_r2", "validation_r2"),
            "Train R2 vs Validation R2",
            "R2",
            config.training_plots_dir / "validation_r2_curve.png",
        )
    else:
        plot_paths["validation_r2_curve"] = plot_single_metric_curve(
            training_log,
            "validation_r2",
            "Validation R2 Curve",
            "R2",
            config.training_plots_dir / "validation_r2_curve.png",
        )

    if "learning_rate" in training_log.columns:
        plot_paths["learning_rate_curve"] = plot_single_metric_curve(
            training_log,
            "learning_rate",
            "Learning Rate Curve",
            "Learning Rate",
            config.training_plots_dir / "learning_rate_curve.png",
            color="#9333ea",
        )
    else:
        skipped_plots.append(
            {
                "plot": "learning_rate_curve",
                "reason": "training_log does not contain learning_rate",
            }
        )

    if "gradient_norm" in training_log.columns:
        plot_paths["gradient_norm_curve"] = plot_single_metric_curve(
            training_log,
            "gradient_norm",
            "Gradient Norm Curve",
            "Gradient Norm",
            config.training_plots_dir / "gradient_norm_curve.png",
            color="#ea580c",
        )
    else:
        skipped_plots.append(
            {
                "plot": "gradient_norm_curve",
                "reason": "training_log does not contain gradient_norm",
            }
        )

    if y_true is not None and y_pred is not None:
        plot_paths["optional_residual_distribution"] = plot_residual_distribution(y_true, y_pred, config)
    else:
        skipped_plots.append(
            {
                "plot": "optional_residual_distribution",
                "reason": "Residual analysis needs y_true and y_pred; main handling is Phase 18/19",
            }
        )

    return plot_paths, skipped_plots


def build_phase_15_summary(plot_paths: dict[str, Path], skipped_plots: list[dict[str, str]]) -> dict[str, pd.DataFrame]:
    """Build display-ready Phase 15 summary tables."""

    plot_table = pd.DataFrame(
        [
            {
                "plot_name": name,
                "saved_path": str(path),
                "status": "created",
            }
            for name, path in plot_paths.items()
        ]
    )
    skipped_table = pd.DataFrame(skipped_plots) if skipped_plots else pd.DataFrame(
        [{"plot": "-", "reason": "No plots skipped"}]
    )
    return {
        "created_plots": plot_table,
        "skipped_optional_plots": skipped_table,
    }


def display_phase_15_summary(summary: dict[str, pd.DataFrame]) -> None:
    """Display Phase 15 summary tables in a notebook, with a plain fallback."""

    sections = [
        ("### Created training plots", "created_plots", "Phase 15 saved training plots"),
        ("### Skipped optional plots", "skipped_optional_plots", "Phase 15 optional plot notes"),
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


def run_phase_15_visualize_training(
    training_log: pd.DataFrame,
    config: ProjectConfig,
    y_true=None,
    y_pred=None,
) -> tuple[dict[str, Path], dict[str, pd.DataFrame]]:
    """Run Phase 15: save training plots and return display-ready summary tables."""

    plot_paths, skipped_plots = build_training_plot_paths(
        training_log=training_log,
        config=config,
        y_true=y_true,
        y_pred=y_pred,
    )
    summary = build_phase_15_summary(plot_paths, skipped_plots)
    return plot_paths, summary
