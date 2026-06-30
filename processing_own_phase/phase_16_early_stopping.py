"""Phase 16: early stopping and best model saving."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd

from .phase_1_import_library import ProjectConfig
from .table_display import style_colored_table as shared_style_colored_table


try:
    import torch
except ModuleNotFoundError:
    torch = None


def require_torch_early_stopping():
    """Return torch or raise a helpful setup error."""

    if torch is None:
        raise ModuleNotFoundError(
            "PyTorch is required for EarlyStopping model saving/restoring. "
            "Use the Jupyter kernel that has torch installed, then restart and rerun the notebook."
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


class EarlyStopping:
    """Track validation loss and save best PyTorch model weights."""

    def __init__(
        self,
        patience: int,
        min_delta: float,
        model_path: str | Path,
        restore_best_model: bool = True,
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.model_path = Path(model_path)
        self.restore_best_model = restore_best_model
        self.best_loss = float("inf")
        self.counter = 0
        self.best_epoch = 0
        self.best_state_dict = None

    def step(self, validation_loss: float, model, epoch: int) -> bool:
        torch_module = require_torch_early_stopping()
        improved = validation_loss < (self.best_loss - self.min_delta)
        if improved:
            self.best_loss = validation_loss
            self.best_epoch = epoch
            self.counter = 0
            self.best_state_dict = copy.deepcopy(model.state_dict())
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            torch_module.save(self.best_state_dict, self.model_path)
            return False
        self.counter += 1
        return self.counter >= self.patience

    def restore(self, model):
        torch_module = require_torch_early_stopping()
        if not self.restore_best_model:
            return model
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)
        elif self.model_path.exists():
            model.load_state_dict(torch_module.load(self.model_path, map_location="cpu"))
        return model


def create_early_stopping(config: ProjectConfig) -> EarlyStopping:
    return EarlyStopping(
        patience=config.patience,
        min_delta=config.min_delta,
        model_path=config.best_model_path,
        restore_best_model=True,
    )


def summarize_early_stopping(early_stopping: EarlyStopping) -> dict[str, object]:
    """Summarize early stopping state after model training."""

    best_model_saved = early_stopping.model_path.exists()
    return {
        "best_loss": early_stopping.best_loss,
        "best_epoch": early_stopping.best_epoch,
        "patience": early_stopping.patience,
        "min_delta": early_stopping.min_delta,
        "counter": early_stopping.counter,
        "model_path": str(early_stopping.model_path),
        "best_model_saved": best_model_saved,
        "restore_best_model": early_stopping.restore_best_model,
        "status": "best model saved" if best_model_saved else "best model file not found",
    }


def save_early_stopping_summary(summary: dict[str, object], config: ProjectConfig) -> Path:
    """Save early stopping summary as JSON."""

    summary_path = config.logs_dir / "early_stopping_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def build_early_stopping_summary_table(summary: dict[str, object]) -> pd.DataFrame:
    """Convert the early stopping summary to a display-friendly table."""

    return pd.DataFrame(
        [
            {"item": key, "value": value}
            for key, value in summary.items()
        ]
    )


def build_early_stopping_rule_table() -> pd.DataFrame:
    """Explain the early stopping decision logic."""

    return pd.DataFrame(
        [
            {
                "condition": "validation_loss improves by at least min_delta",
                "action": "save best_mlp_model.pth and reset counter",
            },
            {
                "condition": "validation_loss does not improve",
                "action": "increase patience counter",
            },
            {
                "condition": "counter reaches patience",
                "action": "stop training and restore best model",
            },
            {
                "condition": "best model exists",
                "action": "Phase 18 should evaluate this restored/best model",
            },
        ]
    )


def build_phase_16_summary(summary: dict[str, object]) -> dict[str, pd.DataFrame]:
    """Build display-ready Phase 16 summary tables."""

    return {
        "early_stopping_summary": build_early_stopping_summary_table(summary),
        "early_stopping_rules": build_early_stopping_rule_table(),
    }


def display_phase_16_summary(summary_tables: dict[str, pd.DataFrame]) -> None:
    """Display Phase 16 summary tables in a notebook, with a plain fallback."""

    sections = [
        ("### Early stopping summary", "early_stopping_summary", "Phase 16 early stopping result"),
        ("### Early stopping rules", "early_stopping_rules", "How early stopping protects validation performance"),
    ]
    try:
        from IPython.display import Markdown, display
    except ModuleNotFoundError:
        for title, key, _caption in sections:
            print(title.replace("#", "").strip())
            print(summary_tables[key])
        return

    for title, key, caption in sections:
        display(Markdown(title))
        display(style_colored_table(summary_tables[key], caption))


def run_phase_16_early_stopping(
    early_stopping: EarlyStopping,
    config: ProjectConfig,
) -> tuple[dict[str, object], dict[str, pd.DataFrame]]:
    """Run Phase 16 reporting for early stopping and best model saving."""

    summary = summarize_early_stopping(early_stopping)
    summary["summary_path"] = str(save_early_stopping_summary(summary, config))
    summary_tables = build_phase_16_summary(summary)
    return summary, summary_tables
