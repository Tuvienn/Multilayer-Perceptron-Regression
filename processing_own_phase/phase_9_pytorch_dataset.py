"""Phase 9: convert processed arrays into PyTorch TensorDataset objects."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .table_display import style_colored_table as shared_style_colored_table


try:
    import torch
    from torch.utils.data import TensorDataset
except ModuleNotFoundError:
    torch = None
    TensorDataset = None


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


def require_torch():
    """Return the torch module or raise a helpful setup error."""

    if torch is None or TensorDataset is None:
        raise ModuleNotFoundError(
            "PyTorch is required from Phase 9 onward. Install torch in the active Jupyter kernel, "
            "then restart the kernel and run the notebook again."
        )
    return torch


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

def create_tensor_dataset(X, y):
    """Create a TensorDataset with float32 features and target shape (n_samples, 1)."""

    torch_module = require_torch()
    X_array = np.asarray(X, dtype=np.float32)
    y_array = np.asarray(y, dtype=np.float32).reshape(-1, 1)
    if X_array.shape[0] != y_array.shape[0]:
        raise ValueError(f"X and y row count mismatch: {X_array.shape[0]} != {y_array.shape[0]}")
    X_tensor = torch_module.tensor(X_array, dtype=torch_module.float32)
    y_tensor = torch_module.tensor(y_array, dtype=torch_module.float32)
    return TensorDataset(X_tensor, y_tensor)


def summarize_tensor_dataset(dataset, split_name: str) -> dict[str, object]:
    """Return shape and dtype details for one TensorDataset."""

    X_tensor, y_tensor = dataset.tensors
    return {
        "split": split_name,
        "samples": len(dataset),
        "X_shape": tuple(X_tensor.shape),
        "y_shape": tuple(y_tensor.shape),
        "X_dtype": str(X_tensor.dtype),
        "y_dtype": str(y_tensor.dtype),
        "target_is_column_vector": len(y_tensor.shape) == 2 and y_tensor.shape[1] == 1,
    }


def build_dataset_summary(train_dataset, val_dataset, test_dataset) -> pd.DataFrame:
    """Summarize all TensorDataset outputs."""

    return pd.DataFrame(
        [
            summarize_tensor_dataset(train_dataset, "train"),
            summarize_tensor_dataset(val_dataset, "validation"),
            summarize_tensor_dataset(test_dataset, "test"),
        ]
    )


def build_tensor_requirement_table() -> pd.DataFrame:
    """Explain why tensors are shaped this way for MLP regression."""

    return pd.DataFrame(
        [
            {
                "component": "Features",
                "requirement": "torch.float32 with shape (n_samples, n_features)",
                "reason": "MLP Linear layers expect numeric floating-point input",
            },
            {
                "component": "Target",
                "requirement": "torch.float32 with shape (n_samples, 1)",
                "reason": "Regression model output layer returns one value per sample",
            },
            {
                "component": "Dataset",
                "requirement": "TensorDataset(X_tensor, y_tensor)",
                "reason": "DataLoader can create batches from paired tensors",
            },
        ]
    )


def build_phase_9_summary(train_dataset, val_dataset, test_dataset) -> dict[str, pd.DataFrame]:
    """Build colored-table friendly Phase 9 summaries."""

    return {
        "dataset_summary": build_dataset_summary(train_dataset, val_dataset, test_dataset),
        "tensor_requirements": build_tensor_requirement_table(),
    }


def display_phase_9_summary(summary: dict[str, pd.DataFrame]) -> None:
    """Display Phase 9 summary tables in a notebook, with a plain fallback."""

    sections = [
        ("### TensorDataset outputs", "dataset_summary", "PyTorch TensorDataset shapes and dtypes"),
        ("### Tensor requirements", "tensor_requirements", "Tensor rules for MLP regression"),
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


def run_phase_9_pytorch_dataset(X_train, y_train, X_val, y_val, X_test, y_test):
    """Create train/validation/test TensorDataset objects."""

    train_dataset = create_tensor_dataset(X_train, y_train)
    val_dataset = create_tensor_dataset(X_val, y_val)
    test_dataset = create_tensor_dataset(X_test, y_test)
    return train_dataset, val_dataset, test_dataset
