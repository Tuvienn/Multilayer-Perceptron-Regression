"""Phase 12: model and training hyperparameter configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass


import pandas as pd

from .table_display import style_colored_table as shared_style_colored_table

from .phase_1_import_library import ProjectConfig


try:
    import torch
except ModuleNotFoundError:
    torch = None


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


@dataclass
class ModelHyperparameters:
    learning_rate: float
    batch_size: int
    epochs: int
    hidden_units: tuple[int, ...]
    dropout: float
    weight_decay: float
    l1_lambda: float
    patience: int
    min_delta: float
    loss_function: str = "mse"
    huber_delta: float = 1.0
    gradient_clip_norm: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


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

def get_model_hyperparameters(config: ProjectConfig) -> ModelHyperparameters:
    """Read model hyperparameters from the central project config."""

    return ModelHyperparameters(
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        epochs=config.epochs,
        hidden_units=config.hidden_units,
        dropout=config.dropout,
        weight_decay=config.weight_decay,
        l1_lambda=config.l1_lambda,
        loss_function=config.loss_function,
        huber_delta=config.huber_delta,
        gradient_clip_norm=config.gradient_clip_norm,
        patience=config.patience,
        min_delta=config.min_delta,
    )


def get_training_device() -> str:
    """Select the training device."""

    if torch is None:
        return "torch-not-installed"
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_hyperparameter_table(hyperparameters: ModelHyperparameters) -> pd.DataFrame:
    """Explain each model hyperparameter."""

    explanations = {
        "learning_rate": "Step size for Adam weight updates",
        "batch_size": "Number of samples per training batch",
        "epochs": "Maximum number of passes over the training set",
        "hidden_units": "MLP hidden layer sizes",
        "dropout": "Dropout rate to reduce overfitting",
        "weight_decay": "L2 regularization strength",
        "l1_lambda": "L1 regularization strength added to the training loss",
        "loss_function": "Regression objective: mse, huber or smooth_l1",
        "huber_delta": "Huber/SmoothL1 transition point for robust regression",
        "gradient_clip_norm": "Optional max gradient norm for stable neural network training",
        "patience": "Early stopping wait epochs",
        "min_delta": "Minimum validation-loss improvement for early stopping",
    }
    rows = []
    for name, value in hyperparameters.to_dict().items():
        rows.append({"hyperparameter": name, "value": str(value), "meaning": explanations.get(name, "")})
    return pd.DataFrame(rows)


def build_training_component_table(device: str) -> pd.DataFrame:
    """Describe the training components used by Phase 13."""

    return pd.DataFrame(
        [
            {
                "component": "device",
                "value": device,
                "purpose": "Run training on GPU if available, otherwise CPU",
            },
            {
                "component": "loss_fn",
                "value": "torch.nn.MSELoss() / HuberLoss() / SmoothL1Loss()",
                "purpose": "Optimize regression error; robust losses reduce sensitivity to high-error outliers",
            },
            {
                "component": "optimizer",
                "value": "torch.optim.Adam()",
                "purpose": "Update weights using adaptive learning rates and L2 weight_decay",
            },
            {
                "component": "regularization",
                "value": "L1 penalty + L2 weight_decay",
                "purpose": "Reduce overfitting and limit the influence of extreme outlier observations",
            },
            {
                "component": "tracked_metrics",
                "value": "MAE, MSE, RMSE, R2",
                "purpose": "Monitor training and validation behavior",
            },
        ]
    )


def build_regularization_strategy_table(hyperparameters: ModelHyperparameters) -> pd.DataFrame:
    """Explain how L1/L2 regularization is used for outlier-sensitive training."""

    return pd.DataFrame(
        [
            {
                "item": "Outlier policy",
                "value": "Flag, analyze, and keep valid house-price outliers",
                "reason": "Very expensive houses may be legitimate observations, not data errors",
            },
            {
                "item": "L1 regularization",
                "value": f"l1_lambda = {hyperparameters.l1_lambda}",
                "reason": "Adds absolute-weight penalty to discourage overly complex weights",
            },
            {
                "item": "L2 regularization",
                "value": f"weight_decay = {hyperparameters.weight_decay}",
                "reason": "Penalizes large weights through Adam weight decay",
            },
            {
                "item": "Target transform",
                "value": "log1p(price)",
                "reason": "Compresses the right-skewed target before model training",
            },
        ]
    )


def build_phase_12_summary(
    hyperparameters: ModelHyperparameters,
    device: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Build colored-table friendly Phase 12 summaries."""

    selected_device = device or get_training_device()
    return {
        "hyperparameters": build_hyperparameter_table(hyperparameters),
        "training_components": build_training_component_table(selected_device),
        "regularization_strategy": build_regularization_strategy_table(hyperparameters),
    }


def display_phase_12_summary(summary: dict[str, pd.DataFrame]) -> None:
    """Display Phase 12 summary tables in a notebook, with a plain fallback."""

    sections = [
        ("### Hyperparameters", "hyperparameters", "Model hyperparameters from ProjectConfig"),
        ("### Training components", "training_components", "Device, loss, optimizer and metrics"),
        ("### Regularization strategy", "regularization_strategy", "L1/L2 strategy for outlier-sensitive training"),
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


def run_phase_12_model_configuration(config: ProjectConfig) -> tuple[ModelHyperparameters, dict[str, pd.DataFrame]]:
    """Create model hyperparameters and display-ready training configuration summary."""

    hyperparameters = get_model_hyperparameters(config)
    summary = build_phase_12_summary(hyperparameters)
    return hyperparameters, summary
