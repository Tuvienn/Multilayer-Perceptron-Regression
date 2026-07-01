"""Phase 19: prediction visualizations and residual diagnostics."""

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
    "actual": "#2563eb",
    "prediction": "#dc2626",
    "residual": "#0891b2",
    "histogram": "#7c3aed",
    "error": "#ea580c",
    "high_end": "#be123c",
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


def validate_predictions_df(predictions_df: pd.DataFrame) -> None:
    """Validate Phase 19 prediction input."""

    required_columns = ["actual_price", "predicted_price"]
    missing_columns = [column for column in required_columns if column not in predictions_df.columns]
    if missing_columns:
        raise ValueError(f"predictions_df is missing required columns: {missing_columns}")
    if predictions_df.empty:
        raise ValueError("predictions_df is empty. Run Phase 18 before Phase 19.")


def add_error_columns(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure residual and error columns exist."""

    validate_predictions_df(predictions_df)
    output_df = predictions_df.copy()
    output_df["residual"] = output_df["actual_price"] - output_df["predicted_price"]
    output_df["absolute_error"] = output_df["residual"].abs()
    output_df["absolute_percentage_error"] = np.where(
        output_df["actual_price"] != 0,
        output_df["absolute_error"] / output_df["actual_price"] * 100,
        np.nan,
    )
    return output_df


def _apply_grid() -> None:
    plt.grid(True, color=PLOT_STYLE["grid"], alpha=0.4, linewidth=0.8)


def plot_actual_vs_predicted(predictions_df: pd.DataFrame, config: ProjectConfig) -> Path:
    actual = predictions_df["actual_price"]
    predicted = predictions_df["predicted_price"]
    min_value = min(actual.min(), predicted.min())
    max_value = max(actual.max(), predicted.max())
    plt.figure(figsize=(7, 7))
    plt.scatter(actual, predicted, alpha=0.35, s=14, color=PLOT_STYLE["actual"])
    plt.plot([min_value, max_value], [min_value, max_value], color=PLOT_STYLE["prediction"], linewidth=2)
    plt.title("Actual Price vs Predicted Price")
    plt.xlabel("Actual price")
    plt.ylabel("Predicted price")
    _apply_grid()
    return _save_plot(config.prediction_plots_dir / "actual_vs_predicted_mlp.png")


def plot_actual_vs_predicted_high_end(
    predictions_df: pd.DataFrame,
    config: ProjectConfig,
    high_end_threshold: float = 1_000_000,
) -> Path:
    """Plot actual vs predicted prices and highlight expensive homes."""

    working_df = add_error_columns(predictions_df)
    high_end_mask = working_df["actual_price"] >= high_end_threshold
    actual = working_df["actual_price"]
    predicted = working_df["predicted_price"]
    min_value = min(actual.min(), predicted.min())
    max_value = max(actual.max(), predicted.max())
    plt.figure(figsize=(7.5, 7))
    plt.scatter(
        working_df.loc[~high_end_mask, "actual_price"],
        working_df.loc[~high_end_mask, "predicted_price"],
        alpha=0.25,
        s=14,
        color=PLOT_STYLE["actual"],
        label=f"Actual < {high_end_threshold:,.0f}",
    )
    plt.scatter(
        working_df.loc[high_end_mask, "actual_price"],
        working_df.loc[high_end_mask, "predicted_price"],
        alpha=0.65,
        s=24,
        color=PLOT_STYLE["high_end"],
        label=f"Actual >= {high_end_threshold:,.0f}",
    )
    plt.plot([min_value, max_value], [min_value, max_value], color=PLOT_STYLE["prediction"], linewidth=2)
    plt.title("Actual vs Predicted Price - High-End Homes Highlighted")
    plt.xlabel("Actual price")
    plt.ylabel("Predicted price")
    plt.legend()
    _apply_grid()
    return _save_plot(config.prediction_plots_dir / "actual_vs_predicted_high_end_mlp.png")


def plot_residuals(predictions_df: pd.DataFrame, config: ProjectConfig) -> Path:
    residuals = predictions_df["actual_price"] - predictions_df["predicted_price"]
    plt.figure(figsize=(9, 5.5))
    plt.scatter(predictions_df["predicted_price"], residuals, alpha=0.35, s=14, color=PLOT_STYLE["residual"])
    plt.axhline(0, color=PLOT_STYLE["prediction"], linewidth=2)
    plt.title("Residual Plot")
    plt.xlabel("Predicted price")
    plt.ylabel("Residual")
    _apply_grid()
    return _save_plot(config.prediction_plots_dir / "residual_plot_mlp.png")


def plot_residual_histogram(predictions_df: pd.DataFrame, config: ProjectConfig) -> Path:
    residuals = predictions_df["actual_price"] - predictions_df["predicted_price"]
    plt.figure(figsize=(9, 5.5))
    plt.hist(residuals, bins=50, color=PLOT_STYLE["histogram"], edgecolor="white")
    plt.axvline(0, color="#111827", linewidth=1.5, linestyle="--")
    plt.title("Residual Histogram")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    _apply_grid()
    return _save_plot(config.prediction_plots_dir / "residual_histogram_mlp.png")


def plot_residual_histogram_clipped(
    predictions_df: pd.DataFrame,
    config: ProjectConfig,
    lower_percentile: float = 1,
    upper_percentile: float = 99,
) -> Path:
    residuals = predictions_df["actual_price"] - predictions_df["predicted_price"]
    residuals = pd.to_numeric(residuals, errors="coerce").dropna()
    lower, upper = np.percentile(residuals, [lower_percentile, upper_percentile])
    clipped_residuals = residuals[(residuals >= lower) & (residuals <= upper)]

    plt.figure(figsize=(9, 5.5))
    plt.hist(clipped_residuals, bins=50, color=PLOT_STYLE["histogram"], edgecolor="white")
    if lower <= 0 <= upper:
        plt.axvline(0, color="#111827", linewidth=1.5, linestyle="--")
    plt.title("Residual Histogram Clipped to 1st-99th Percentiles")
    plt.xlabel("Residual = actual price - predicted price")
    plt.ylabel("Frequency")
    _apply_grid()
    return _save_plot(config.prediction_plots_dir / "residual_histogram_clipped_1_99.png")


def build_error_by_price_range(predictions_df: pd.DataFrame, bins: int = 5) -> pd.DataFrame:
    """Aggregate absolute errors by actual price range."""

    working_df = add_error_columns(predictions_df)
    working_df["price_range"] = pd.qcut(
        working_df["actual_price"],
        q=bins,
        duplicates="drop",
    )
    summary = (
        working_df.groupby("price_range", observed=True)
        .agg(
            samples=("actual_price", "size"),
            mean_actual_price=("actual_price", "mean"),
            mean_absolute_error=("absolute_error", "mean"),
            median_absolute_error=("absolute_error", "median"),
            mean_absolute_percentage_error=("absolute_percentage_error", "mean"),
        )
        .reset_index()
    )
    summary["price_range"] = summary["price_range"].astype(str)
    return summary


def calculate_prediction_metrics(predictions_df: pd.DataFrame) -> dict[str, float]:
    """Calculate regression metrics for a prediction subset."""

    y_true = predictions_df["actual_price"].to_numpy(dtype=float)
    y_pred = predictions_df["predicted_price"].to_numpy(dtype=float)
    residuals = y_true - y_pred
    mae = float(np.mean(np.abs(residuals)))
    mse = float(np.mean(residuals**2))
    rmse = float(np.sqrt(mse))
    denominator = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1 - np.sum(residuals**2) / denominator) if denominator > 0 else float("nan")
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


def build_high_end_house_analysis(
    predictions_df: pd.DataFrame,
    quantiles: tuple[float, ...] = (0.90, 0.95),
    fixed_thresholds: tuple[float, ...] = (1_000_000, 2_000_000),
) -> pd.DataFrame:
    """Analyze prediction quality on expensive houses separately."""

    working_df = add_error_columns(predictions_df)
    threshold_specs: list[tuple[str, float]] = []
    for quantile in quantiles:
        threshold = float(working_df["actual_price"].quantile(quantile))
        threshold_specs.append((f"actual_price >= p{int(quantile * 100)}", threshold))
    for threshold in fixed_thresholds:
        threshold_specs.append((f"actual_price >= {int(threshold):,}", float(threshold)))

    rows = []
    for segment, threshold in threshold_specs:
        segment_df = working_df[working_df["actual_price"] >= threshold].copy()
        if segment_df.empty:
            continue
        metrics = calculate_prediction_metrics(segment_df)
        rows.append(
            {
                "segment": segment,
                "threshold": threshold,
                "samples": len(segment_df),
                "sample_ratio": len(segment_df) / len(working_df),
                "mean_actual_price": float(segment_df["actual_price"].mean()),
                "mean_predicted_price": float(segment_df["predicted_price"].mean()),
                "mean_residual": float(segment_df["residual"].mean()),
                "underprediction_rate": float((segment_df["residual"] > 0).mean()),
                "median_absolute_error": float(segment_df["absolute_error"].median()),
                "max_absolute_error": float(segment_df["absolute_error"].max()),
                "mean_absolute_percentage_error": float(segment_df["absolute_percentage_error"].mean()),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def plot_high_end_error_metrics(high_end_house_analysis: pd.DataFrame, config: ProjectConfig) -> Path | None:
    """Plot RMSE by expensive-house segment."""

    if high_end_house_analysis.empty or "RMSE" not in high_end_house_analysis.columns:
        return None
    plot_df = high_end_house_analysis.copy()
    plt.figure(figsize=(10, 5.5))
    plt.bar(plot_df["segment"], plot_df["RMSE"], color=PLOT_STYLE["high_end"])
    plt.title("MLP Error on High-End House Segments")
    plt.xlabel("High-end segment")
    plt.ylabel("RMSE")
    plt.xticks(rotation=20, ha="right")
    _apply_grid()
    return _save_plot(config.prediction_plots_dir / "high_end_house_rmse_mlp.png")


def plot_error_by_price_range(predictions_df: pd.DataFrame, config: ProjectConfig) -> tuple[Path, pd.DataFrame]:
    """Plot model error by actual price segment."""

    summary = build_error_by_price_range(predictions_df)
    plt.figure(figsize=(10, 5.5))
    plt.bar(summary["price_range"], summary["mean_absolute_error"], color=PLOT_STYLE["error"])
    plt.title("Error by Actual Price Range")
    plt.xlabel("Actual price range")
    plt.ylabel("Mean absolute error")
    plt.xticks(rotation=20, ha="right")
    _apply_grid()
    return _save_plot(config.prediction_plots_dir / "error_by_price_range_mlp.png"), summary


def build_top_error_table(predictions_df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Return top prediction errors."""

    working_df = add_error_columns(predictions_df)
    return working_df.sort_values("absolute_error", ascending=False).head(top_n).reset_index(drop=True)


def plot_top_largest_errors(predictions_df: pd.DataFrame, config: ProjectConfig, top_n: int = 20) -> tuple[Path, pd.DataFrame]:
    """Plot the largest absolute prediction errors."""

    top_errors = build_top_error_table(predictions_df, top_n=top_n)
    top_errors = top_errors.copy()
    top_errors["case"] = [f"#{index + 1}" for index in range(len(top_errors))]
    plt.figure(figsize=(9, 7))
    plt.barh(top_errors["case"], top_errors["absolute_error"], color="#be123c")
    plt.gca().invert_yaxis()
    plt.title(f"Top {top_n} Largest MLP Prediction Errors")
    plt.xlabel("Absolute error")
    plt.ylabel("Case rank")
    _apply_grid()
    return _save_plot(config.prediction_plots_dir / "top_20_largest_errors_mlp.png"), top_errors


def plot_training_loss_curve(training_log: pd.DataFrame, config: ProjectConfig) -> Path:
    """Plot train and validation loss for prediction-context reporting."""

    required_columns = ["epoch", "train_loss", "validation_loss"]
    missing_columns = [column for column in required_columns if column not in training_log.columns]
    if missing_columns:
        raise ValueError(f"training_log is missing columns for loss curve: {missing_columns}")
    plt.figure(figsize=(9, 5.5))
    plt.plot(training_log["epoch"], training_log["train_loss"], label="train_loss", color="#2563eb", linewidth=2)
    plt.plot(training_log["epoch"], training_log["validation_loss"], label="validation_loss", color="#dc2626", linewidth=2)
    target_scale = "log-transformed target" if getattr(config, "log_transform_target", False) else "target"
    plt.title(f"Training and Validation MSE Loss on {target_scale}")
    plt.xlabel("Epoch")
    plt.ylabel(f"MSE Loss on {target_scale}")
    plt.legend()
    _apply_grid()
    return _save_plot(config.prediction_plots_dir / "training_validation_loss_for_predictions.png")


def build_phase_19_summary(
    plot_paths: dict[str, Path],
    error_by_price_range: pd.DataFrame,
    top_errors: pd.DataFrame,
    high_end_house_analysis: pd.DataFrame,
    skipped_plots: list[dict[str, str]],
) -> dict[str, pd.DataFrame]:
    """Build display-ready Phase 19 summary tables."""

    return {
        "created_plots": pd.DataFrame(
            [{"plot_name": name, "saved_path": str(path), "status": "created"} for name, path in plot_paths.items()]
        ),
        "error_by_price_range": error_by_price_range,
        "high_end_house_analysis": high_end_house_analysis,
        "top_20_largest_errors": top_errors,
        "skipped_plots": pd.DataFrame(skipped_plots) if skipped_plots else pd.DataFrame(
            [{"plot": "-", "reason": "No plots skipped"}]
        ),
    }


def display_phase_19_summary(summary: dict[str, pd.DataFrame]) -> None:
    """Display Phase 19 summary tables in a notebook, with a plain fallback."""

    sections = [
        ("### Created prediction plots", "created_plots", "Phase 19 saved prediction plots"),
        ("### Error by price range", "error_by_price_range", "Prediction error by actual price range"),
        ("### High-end house analysis", "high_end_house_analysis", "Prediction error for expensive homes"),
        ("### Top 20 largest errors", "top_20_largest_errors", "Largest MLP prediction errors"),
        ("### Skipped plots", "skipped_plots", "Phase 19 optional plot notes"),
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


def run_phase_19_predictions_visualization(
    predictions_df: pd.DataFrame,
    config: ProjectConfig,
    training_log: pd.DataFrame | None = None,
) -> tuple[dict[str, Path], dict[str, pd.DataFrame]]:
    """Run Phase 19 prediction visualizations."""

    predictions_df = add_error_columns(predictions_df)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(config.predictions_path, index=False)
    top_errors_path = config.results_dir / "top_20_largest_prediction_errors.csv"
    high_end_analysis_path = config.results_dir / "high_end_house_error_analysis.csv"

    error_plot_path, error_by_price_range = plot_error_by_price_range(predictions_df, config)
    top_error_plot_path, top_errors = plot_top_largest_errors(predictions_df, config)
    high_end_house_analysis = build_high_end_house_analysis(predictions_df)
    high_end_plot_path = plot_high_end_error_metrics(high_end_house_analysis, config)
    top_errors.to_csv(top_errors_path, index=False)
    high_end_house_analysis.to_csv(high_end_analysis_path, index=False)

    plot_paths = {
        "actual_vs_predicted": plot_actual_vs_predicted(predictions_df, config),
        "actual_vs_predicted_high_end": plot_actual_vs_predicted_high_end(predictions_df, config),
        "residual_plot": plot_residuals(predictions_df, config),
        "residual_histogram": plot_residual_histogram(predictions_df, config),
        "residual_histogram_clipped_1_99": plot_residual_histogram_clipped(predictions_df, config),
        "error_by_price_range": error_plot_path,
        "top_20_largest_errors": top_error_plot_path,
    }
    if high_end_plot_path is not None:
        plot_paths["high_end_house_rmse"] = high_end_plot_path
    skipped_plots: list[dict[str, str]] = []
    if training_log is not None and not training_log.empty:
        plot_paths["training_validation_loss"] = plot_training_loss_curve(training_log, config)
    else:
        skipped_plots.append(
            {
                "plot": "training_validation_loss",
                "reason": "training_log was not provided; Phase 15 already contains the main training curves",
            }
        )

    summary = build_phase_19_summary(
        plot_paths,
        error_by_price_range,
        top_errors,
        high_end_house_analysis,
        skipped_plots,
    )
    return plot_paths, summary
