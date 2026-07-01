"""Phase 10: create PyTorch DataLoader objects."""

from __future__ import annotations

import math

import pandas as pd
from .table_display import style_colored_table as shared_style_colored_table
from .phase_1_import_library import ProjectConfig


try:
    # pyrefly: ignore [missing-import]
    import torch
    from torch.utils.data import DataLoader
except ModuleNotFoundError:
    torch = None
    DataLoader = None


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


def require_torch_dataloader():
    """Return DataLoader or raise a helpful setup error."""

    if torch is None or DataLoader is None:
        raise ModuleNotFoundError(
            "PyTorch is required from Phase 10 onward. Install torch in the active Jupyter kernel, "
            "then restart the kernel and run the notebook again."
        )
    return torch, DataLoader


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

def create_data_loader(dataset, batch_size: int, shuffle: bool, random_state: int | None = None):
    """Create one DataLoader with optional deterministic shuffling."""

    torch_module, data_loader_class = require_torch_dataloader()
    generator = None
    if shuffle and random_state is not None:
        generator = torch_module.Generator()
        generator.manual_seed(random_state)
    return data_loader_class(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def summarize_data_loader(loader, split_name: str, shuffle: bool) -> dict[str, object]:
    """Return batch details for one DataLoader."""

    dataset_size = len(loader.dataset)
    batch_size = loader.batch_size
    expected_batches = math.ceil(dataset_size / batch_size)
    first_batch = next(iter(loader))
    X_batch, y_batch = first_batch
    return {
        "split": split_name,
        "samples": dataset_size,
        "batch_size": batch_size,
        "batches": len(loader),
        "expected_batches": expected_batches,
        "shuffle": shuffle,
        "first_X_batch_shape": tuple(X_batch.shape),
        "first_y_batch_shape": tuple(y_batch.shape),
    }


def build_dataloader_summary(train_loader, val_loader, test_loader) -> pd.DataFrame:
    """Summarize train/validation/test DataLoader objects."""

    return pd.DataFrame(
        [
            summarize_data_loader(train_loader, "train", True),
            summarize_data_loader(val_loader, "validation", False),
            summarize_data_loader(test_loader, "test", False),
        ]
    )


def build_dataloader_rule_table() -> pd.DataFrame:
    """Explain shuffle choices for each split."""

    return pd.DataFrame(
        [
            {
                "split": "train",
                "shuffle": True,
                "reason": "Random batches reduce order bias during gradient updates",
            },
            {
                "split": "validation",
                "shuffle": False,
                "reason": "Evaluation does not update weights, so fixed order is easier to reproduce",
            },
            {
                "split": "test",
                "shuffle": False,
                "reason": "Final evaluation should be deterministic",
            },
        ]
    )


def build_phase_10_summary(train_loader, val_loader, test_loader) -> dict[str, pd.DataFrame]:
    """Build colored-table friendly Phase 10 summaries."""

    return {
        "dataloader_summary": build_dataloader_summary(train_loader, val_loader, test_loader),
        "dataloader_rules": build_dataloader_rule_table(),
    }


def display_phase_10_summary(summary: dict[str, pd.DataFrame]) -> None:
    """Display Phase 10 summary tables in a notebook, with a plain fallback."""

    sections = [
        ("### DataLoader outputs", "dataloader_summary", "PyTorch DataLoader batch summary"),
        ("### Shuffle rules", "dataloader_rules", "Why train is shuffled and validation/test are not"),
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


def run_phase_10_pytorch_dataloader(train_dataset, val_dataset, test_dataset, config: ProjectConfig):
    """Create train/validation/test DataLoader objects."""

    train_loader = create_data_loader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        random_state=config.random_state,
    )
    val_loader = create_data_loader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = create_data_loader(test_dataset, batch_size=config.batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
