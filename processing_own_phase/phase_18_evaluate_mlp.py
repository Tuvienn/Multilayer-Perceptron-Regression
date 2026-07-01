"""Phase 18: evaluate the best MLP model on the test set."""

from __future__ import annotations

import time

from typing import Any, Callable

import numpy as np
import pandas as pd

from .phase_1_import_library import ProjectConfig
from .table_display import style_colored_table as shared_style_colored_table


try:
    import torch
except ModuleNotFoundError:
    torch = None


DEFAULT_INFERENCE_REPEATS = 30


def require_torch_evaluation():
    """Return torch or raise a helpful setup error."""

    if torch is None:
        raise ModuleNotFoundError(
            "PyTorch is required for Phase 18 MLP evaluation. "
            "Use the Jupyter kernel that has torch installed, then restart and rerun the notebook."
        )
    return torch


def validate_inference_repeats(n_repeats: int) -> int:
    repeats = int(n_repeats)
    if repeats <= 0:
        raise ValueError("n_repeats must be a positive integer for inference timing.")
    return repeats


def synchronize_device(torch_module, selected_device) -> None:
    if getattr(selected_device, "type", None) == "cuda" and torch_module.cuda.is_available():
        torch_module.cuda.synchronize(selected_device)


def predict_loader_once(model, test_loader, selected_device, torch_module) -> tuple[np.ndarray, np.ndarray]:
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch_module.no_grad():
        for X_batch, y_batch in test_loader:
            output = model(X_batch.to(selected_device))
            predictions.append(output.detach().cpu().numpy())
            targets.append(y_batch.detach().cpu().numpy())
    return np.vstack(targets).reshape(-1), np.vstack(predictions).reshape(-1)


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


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Calculate core regression metrics on the provided target scale."""

    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    residuals = y_true - y_pred
    mae = float(np.mean(np.abs(residuals)))
    mse = float(np.mean(residuals**2))
    rmse = float(np.sqrt(mse))
    denominator = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1 - np.sum(residuals**2) / denominator) if denominator > 0 else float("nan")
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


def predict_mlp_with_timing_details(
    model,
    test_loader,
    device: str | Any | None = None,
    n_repeats: int = DEFAULT_INFERENCE_REPEATS,
) -> tuple[np.ndarray, np.ndarray, float, float, str]:
    """Predict repeatedly and return targets, predictions, mean/std inference time and device."""

    torch_module = require_torch_evaluation()
    repeats = validate_inference_repeats(n_repeats)
    selected_device = torch_module.device(device or ("cuda" if torch_module.cuda.is_available() else "cpu"))
    model.to(selected_device)
    model.eval()

    timings: list[float] = []
    first_y_true: np.ndarray | None = None
    first_y_pred: np.ndarray | None = None
    for repeat_index in range(repeats):
        synchronize_device(torch_module, selected_device)
        inference_start = time.perf_counter()
        y_true, y_pred = predict_loader_once(model, test_loader, selected_device, torch_module)
        synchronize_device(torch_module, selected_device)
        timings.append(time.perf_counter() - inference_start)
        if repeat_index == 0:
            first_y_true = y_true
            first_y_pred = y_pred

    timing_array = np.asarray(timings, dtype=float)
    inference_time_mean = float(timing_array.mean())
    inference_time_std = float(timing_array.std(ddof=1)) if repeats > 1 else 0.0
    if first_y_true is None or first_y_pred is None:
        raise RuntimeError("No predictions were produced during MLP inference timing.")
    return first_y_true, first_y_pred, inference_time_mean, inference_time_std, str(selected_device)


def predict_mlp(
    model,
    test_loader,
    device: str | Any | None = None,
    n_repeats: int = DEFAULT_INFERENCE_REPEATS,
) -> tuple[np.ndarray, np.ndarray, float, str]:
    """Predict with the MLP model and return targets, predictions, mean inference time and device."""

    y_true, y_pred, inference_time_mean, _inference_time_std, selected_device = predict_mlp_with_timing_details(
        model,
        test_loader,
        device=device,
        n_repeats=n_repeats,
    )
    return y_true, y_pred, inference_time_mean, selected_device


def build_predictions_dataframe(y_true, y_pred) -> pd.DataFrame:
    """Create a prediction table with residual diagnostics."""

    y_true_array = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred_array = np.asarray(y_pred, dtype=float).reshape(-1)
    residuals = y_true_array - y_pred_array
    predictions_df = pd.DataFrame(
        {
            "actual_price": y_true_array,
            "predicted_price": y_pred_array,
            "residual": residuals,
            "absolute_error": np.abs(residuals),
        }
    )
    predictions_df["absolute_percentage_error"] = np.where(
        predictions_df["actual_price"] != 0,
        predictions_df["absolute_error"] / predictions_df["actual_price"] * 100,
        np.nan,
    )
    return predictions_df


def build_metric_explanation_table(target_scale: str) -> pd.DataFrame:
    """Explain Phase 18 metrics for notebook reporting."""

    return pd.DataFrame(
        [
            {"metric": "MAE", "meaning": "Average absolute prediction error", "target_scale": target_scale},
            {"metric": "MSE", "meaning": "Average squared prediction error", "target_scale": target_scale},
            {"metric": "RMSE", "meaning": "Square root of MSE, easier to read than MSE", "target_scale": target_scale},
            {"metric": "R2", "meaning": "How much target variation the model explains", "target_scale": target_scale},
        ]
    )


def build_prediction_error_summary(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize prediction errors on the original reporting scale."""

    return pd.DataFrame(
        [
            {
                "rows": len(predictions_df),
                "mean_actual_price": float(predictions_df["actual_price"].mean()),
                "mean_predicted_price": float(predictions_df["predicted_price"].mean()),
                "mean_residual": float(predictions_df["residual"].mean()),
                "median_absolute_error": float(predictions_df["absolute_error"].median()),
                "max_absolute_error": float(predictions_df["absolute_error"].max()),
                "negative_predictions": int((predictions_df["predicted_price"] < 0).sum()),
            }
        ]
    )


def build_phase_18_summary(
    metrics_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    target_scale: str,
) -> dict[str, pd.DataFrame]:
    """Build display-ready Phase 18 summary tables."""

    return {
        "mlp_metrics": metrics_df,
        "metric_explanation": build_metric_explanation_table(target_scale),
        "prediction_error_summary": build_prediction_error_summary(predictions_df),
        "prediction_preview": predictions_df.head(10),
    }


def display_phase_18_summary(summary: dict[str, pd.DataFrame]) -> None:
    """Display Phase 18 summary tables in a notebook, with a plain fallback."""

    sections = [
        ("### MLP test metrics", "mlp_metrics", "Phase 18 MLP metrics on test set"),
        ("### Metric explanation", "metric_explanation", "How to read Phase 18 metrics"),
        ("### Prediction error summary", "prediction_error_summary", "Test prediction error summary"),
        ("### Prediction preview", "prediction_preview", "First 10 MLP predictions"),
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


def run_phase_18_evaluate_mlp(
    model,
    test_loader,
    config: ProjectConfig,
    inverse_transform: Callable | None = None,
    n_repeats: int = DEFAULT_INFERENCE_REPEATS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Run Phase 18 MLP evaluation on the test set."""

    y_true, y_pred, inference_time_mean, inference_time_std, device = predict_mlp_with_timing_details(
        model,
        test_loader,
        n_repeats=n_repeats,
    )
    target_scale = "model target scale"
    if inverse_transform is not None:
        y_true = inverse_transform(y_true)
        y_pred = inverse_transform(y_pred)
        target_scale = "original price scale after expm1 inverse transform"
    metrics_df = pd.DataFrame(
        [
            {
                "model": "MLP Regression PyTorch",
                **regression_metrics(y_true, y_pred),
                "training_time_seconds": pd.NA,
                "inference_time_seconds": inference_time_mean,
                "inference_time_mean_seconds": inference_time_mean,
                "inference_time_std_seconds": inference_time_std,
                "inference_time_repeats": int(n_repeats),
                "target_scale": target_scale,
                "device": device,
            }
        ]
    )
    predictions_df = build_predictions_dataframe(y_true, y_pred)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(config.mlp_metrics_path, index=False)
    predictions_df.to_csv(config.predictions_path, index=False)
    summary = build_phase_18_summary(metrics_df, predictions_df, target_scale)
    return metrics_df, predictions_df, summary
