"""Phase 11: build the PyTorch MLP regression model."""

from __future__ import annotations

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


def require_torch_nn():
    """Return torch and nn or raise a helpful setup error."""

    if torch is None or nn is None:
        raise ModuleNotFoundError(
            "PyTorch is required to build the MLP model. Install torch in the active Jupyter kernel, "
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

if nn is not None:

    class MLPRegression(nn.Module):
        """Feed-forward neural network for house price regression."""

        def __init__(
            self,
            input_dim: int,
            hidden_units: tuple[int, ...] = (128, 64),
            dropout: float = 0.10,
        ) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            previous_dim = input_dim
            for hidden_dim in hidden_units:
                layers.append(nn.Linear(previous_dim, hidden_dim))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                previous_dim = hidden_dim
            layers.append(nn.Linear(previous_dim, 1))
            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)

else:

    class MLPRegression:
        """Placeholder used when PyTorch is not installed."""

        def __init__(self, *args, **kwargs) -> None:
            require_torch_nn()


def build_mlp_model(input_dim: int, hidden_units: tuple[int, ...] = (128, 64), dropout: float = 0.10) -> MLPRegression:
    """Build the MLP regression model."""

    require_torch_nn()
    if input_dim <= 0:
        raise ValueError("input_dim must be positive")
    return MLPRegression(input_dim=input_dim, hidden_units=hidden_units, dropout=dropout)


def count_trainable_parameters(model) -> int:
    """Count trainable model parameters."""

    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def build_model_architecture_table(
    input_dim: int,
    hidden_units: tuple[int, ...],
    dropout: float,
) -> pd.DataFrame:
    """Describe the MLP architecture."""

    rows = [{"layer": "Input", "units": input_dim, "activation": "-", "purpose": "Receive processed features"}]
    for index, hidden_dim in enumerate(hidden_units, start=1):
        rows.append(
            {
                "layer": f"Hidden {index}",
                "units": hidden_dim,
                "activation": "ReLU",
                "purpose": "Learn nonlinear feature interactions",
            }
        )
        if dropout > 0:
            rows.append(
                {
                    "layer": f"Dropout {index}",
                    "units": dropout,
                    "activation": "-",
                    "purpose": "Reduce overfitting during training",
                }
            )
    rows.append({"layer": "Output", "units": 1, "activation": "Linear", "purpose": "Predict one regression value"})
    return pd.DataFrame(rows)


def build_model_summary(model, input_dim: int, hidden_units: tuple[int, ...], dropout: float) -> dict[str, pd.DataFrame]:
    """Build colored-table friendly model summaries."""

    return {
        "architecture": build_model_architecture_table(input_dim, hidden_units, dropout),
        "model_summary": pd.DataFrame(
            [
                {
                    "model": model.__class__.__name__,
                    "input_dim": input_dim,
                    "hidden_layers": len(hidden_units),
                    "hidden_units": str(hidden_units),
                    "dropout": dropout,
                    "output_dim": 1,
                    "trainable_parameters": count_trainable_parameters(model),
                }
            ]
        ),
    }


def display_phase_11_summary(summary: dict[str, pd.DataFrame]) -> None:
    """Display Phase 11 summary tables in a notebook, with a plain fallback."""

    sections = [
        ("### MLP architecture", "architecture", "MLP regression architecture"),
        ("### Model summary", "model_summary", "Trainable model summary"),
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


def run_phase_11_build_mlp_model(
    input_dim: int,
    hidden_units: tuple[int, ...] = (128, 64),
    dropout: float = 0.10,
) -> tuple[MLPRegression, dict[str, pd.DataFrame]]:
    """Build the MLP model and return display-ready summary tables."""

    model = build_mlp_model(input_dim, hidden_units, dropout)
    summary = build_model_summary(model, input_dim, hidden_units, dropout)
    return model, summary
