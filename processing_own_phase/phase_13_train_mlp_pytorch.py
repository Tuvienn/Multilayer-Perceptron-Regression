"""Phase 13: PyTorch training loop for MLP regression."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from .table_display import style_colored_table as shared_style_colored_table


try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None


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


def require_torch_training():
    """Return torch and nn or raise a helpful setup error."""

    if torch is None or nn is None:
        raise ModuleNotFoundError(
            "PyTorch is required to train the MLP model. Install torch in the active Jupyter kernel, "
            "then restart the kernel and run the notebook again."
        )
    return torch, nn


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

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate regression metrics on the current target scale."""

    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    residuals = y_true - y_pred
    mae = float(np.mean(np.abs(residuals)))
    mse = float(np.mean(residuals**2))
    rmse = float(np.sqrt(mse))
    denominator = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1 - np.sum(residuals**2) / denominator) if denominator > 0 else float("nan")
    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}


def get_hyperparameter_value(hyperparameters, name: str, default):
    """Read a hyperparameter from a dataclass-like object or dict."""

    if isinstance(hyperparameters, dict):
        return hyperparameters.get(name, default)
    return getattr(hyperparameters, name, default)


def create_regression_loss(hyperparameters):
    """Create the configured PyTorch regression loss."""

    _torch_module, nn_module = require_torch_training()
    loss_function = str(get_hyperparameter_value(hyperparameters, "loss_function", "mse")).lower()
    huber_delta = float(get_hyperparameter_value(hyperparameters, "huber_delta", 1.0))
    if loss_function == "mse":
        return nn_module.MSELoss(), "mse", huber_delta
    if loss_function == "huber":
        if hasattr(nn_module, "HuberLoss"):
            return nn_module.HuberLoss(delta=huber_delta), "huber", huber_delta
        return nn_module.SmoothL1Loss(beta=huber_delta), "smooth_l1_fallback_for_huber", huber_delta
    if loss_function == "smooth_l1":
        try:
            return nn_module.SmoothL1Loss(beta=huber_delta), "smooth_l1", huber_delta
        except TypeError:
            return nn_module.SmoothL1Loss(), "smooth_l1", huber_delta
    raise ValueError("loss_function must be one of: mse, huber, smooth_l1")


def evaluate_loader(model, data_loader, criterion, device) -> tuple[float, dict[str, float]]:
    """Evaluate model without updating gradients."""

    torch_module, _nn_module = require_torch_training()
    model.eval()
    losses: list[float] = []
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch_module.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            output = model(X_batch)
            loss = criterion(output, y_batch)
            losses.append(float(loss.item()))
            predictions.append(output.detach().cpu().numpy())
            targets.append(y_batch.detach().cpu().numpy())
    y_pred = np.vstack(predictions)
    y_true = np.vstack(targets)
    return float(np.mean(losses)), calculate_metrics(y_true, y_pred)


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    gradient_clip_norm: float | None = None,
) -> tuple[float, dict[str, float], float]:
    """Train one epoch and return loss, metrics, and average gradient norm."""

    torch_module, _nn_module = require_torch_training()
    model.train()
    losses: list[float] = []
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    gradient_norms: list[float] = []

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        if gradient_clip_norm is not None:
            total_norm = torch_module.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            gradient_norms.append(float(total_norm))
        else:
            total_norm = 0.0
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter_norm = parameter.grad.detach().data.norm(2)
                    total_norm += float(parameter_norm.item() ** 2)
            gradient_norms.append(total_norm**0.5)
        optimizer.step()

        losses.append(float(loss.item()))
        predictions.append(output.detach().cpu().numpy())
        targets.append(y_batch.detach().cpu().numpy())

    y_pred = np.vstack(predictions)
    y_true = np.vstack(targets)
    return float(np.mean(losses)), calculate_metrics(y_true, y_pred), float(np.mean(gradient_norms))


def train_mlp_pytorch(
    model,
    train_loader,
    val_loader,
    hyperparameters,
    early_stopping: Any | None = None,
    device: str | Any | None = None,
) -> pd.DataFrame:
    """Train the MLP with a standard PyTorch training loop."""

    torch_module, nn_module = require_torch_training()
    selected_device = torch_module.device(device or ("cuda" if torch_module.cuda.is_available() else "cpu"))
    model.to(selected_device)
    criterion, loss_function, huber_delta = create_regression_loss(hyperparameters)
    gradient_clip_norm = get_hyperparameter_value(hyperparameters, "gradient_clip_norm", None)
    optimizer = torch_module.optim.Adam(
        model.parameters(),
        lr=hyperparameters.learning_rate,
        weight_decay=hyperparameters.weight_decay,
    )

    logs: list[dict[str, float | int | str | bool]] = []
    stopped_early = False
    for epoch in range(1, hyperparameters.epochs + 1):
        start_time = time.perf_counter()
        train_loss, train_metrics, gradient_norm = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            selected_device,
            gradient_clip_norm=gradient_clip_norm,
        )
        val_loss, val_metrics = evaluate_loader(model, val_loader, criterion, selected_device)
        epoch_time = time.perf_counter() - start_time
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "train_mae": train_metrics["mae"],
            "train_mse": train_metrics["mse"],
            "train_rmse": train_metrics["rmse"],
            "train_r2": train_metrics["r2"],
            "validation_mae": val_metrics["mae"],
            "validation_mse": val_metrics["mse"],
            "validation_rmse": val_metrics["rmse"],
            "validation_r2": val_metrics["r2"],
            "learning_rate": hyperparameters.learning_rate,
            "loss_function": loss_function,
            "huber_delta": huber_delta,
            "gradient_clip_norm": gradient_clip_norm,
            "gradient_norm": gradient_norm,
            "epoch_time_seconds": epoch_time,
            "device": str(selected_device),
            "stopped_early": False,
        }
        logs.append(row)
        if early_stopping is not None and early_stopping.step(val_loss, model, epoch):
            stopped_early = True
            logs[-1]["stopped_early"] = True
            break

    if early_stopping is not None:
        early_stopping.restore(model)
    training_log = pd.DataFrame(logs)
    if stopped_early and not training_log.empty:
        training_log.loc[training_log.index[-1], "stopped_early"] = True
    return training_log


def build_training_process_table() -> pd.DataFrame:
    """Explain the PyTorch training loop steps."""

    return pd.DataFrame(
        [
            {"step": "model.train()", "purpose": "Enable training behavior such as dropout"},
            {"step": "forward pass", "purpose": "Predict y for each batch"},
            {"step": "loss = configured_loss(output, y)", "purpose": "Measure regression error with MSE or a robust loss"},
            {"step": "optimizer.zero_grad()", "purpose": "Clear old gradients"},
            {"step": "loss.backward()", "purpose": "Compute gradients"},
            {"step": "optional gradient clipping", "purpose": "Limit gradient norm when configured for robust-loss trials"},
            {"step": "optimizer.step()", "purpose": "Update model weights"},
            {"step": "model.eval() + torch.no_grad()", "purpose": "Validate without updating weights"},
            {"step": "append training log", "purpose": "Store loss and metrics for later monitoring"},
        ]
    )


def build_training_log_summary(training_log: pd.DataFrame) -> pd.DataFrame:
    """Summarize the Phase 13 training result."""

    if training_log.empty:
        return pd.DataFrame(
            [
                {
                    "epochs_ran": 0,
                    "best_epoch": None,
                    "best_validation_loss": None,
                    "final_train_loss": None,
                    "final_validation_loss": None,
                    "stopped_early": False,
                }
            ]
        )
    best_row = training_log.loc[training_log["validation_loss"].idxmin()]
    final_row = training_log.iloc[-1]
    return pd.DataFrame(
        [
            {
                "epochs_ran": int(training_log["epoch"].max()),
                "best_epoch": int(best_row["epoch"]),
                "best_validation_loss": float(best_row["validation_loss"]),
                "final_train_loss": float(final_row["train_loss"]),
                "final_validation_loss": float(final_row["validation_loss"]),
                "final_validation_rmse": float(final_row["validation_rmse"]),
                "stopped_early": bool(final_row.get("stopped_early", False)),
            }
        ]
    )


def build_metric_scale_table(log_transform_target: bool = True) -> pd.DataFrame:
    """Explain the target scale used by training metrics."""

    if log_transform_target:
        metric_scale = "log1p(price)"
        note = "Phase 18 should inverse-transform predictions with expm1 for real-price metrics"
    else:
        metric_scale = "price"
        note = "Training metrics are already on the original price scale"
    return pd.DataFrame(
        [
            {
                "metric_location": "Phase 13 training log",
                "scale": metric_scale,
                "note": note,
            }
        ]
    )


def build_phase_13_summary(training_log: pd.DataFrame, log_transform_target: bool = True) -> dict[str, pd.DataFrame]:
    """Build colored-table friendly Phase 13 summaries."""

    return {
        "training_process": build_training_process_table(),
        "training_log_summary": build_training_log_summary(training_log),
        "metric_scale": build_metric_scale_table(log_transform_target),
        "training_log_tail": training_log.tail(),
    }


def display_phase_13_summary(summary: dict[str, pd.DataFrame]) -> None:
    """Display Phase 13 summary tables in a notebook, with a plain fallback."""

    sections = [
        ("### Training loop steps", "training_process", "PyTorch training loop"),
        ("### Training result", "training_log_summary", "Phase 13 training summary"),
        ("### Metric scale", "metric_scale", "Training-log metric scale"),
        ("### Training log tail", "training_log_tail", "Last epochs in the training log"),
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


def run_phase_13_train_mlp_pytorch(
    model,
    train_loader,
    val_loader,
    hyperparameters,
    config,
    device: str | Any | None = None,
):
    """Run Phase 13 training with early stopping configured internally."""

    require_torch_training()
    from .phase_16_early_stopping import create_early_stopping

    early_stopping = create_early_stopping(config)
    training_log = train_mlp_pytorch(
        model,
        train_loader,
        val_loader,
        hyperparameters,
        early_stopping=early_stopping,
        device=device,
    )
    return training_log, early_stopping
