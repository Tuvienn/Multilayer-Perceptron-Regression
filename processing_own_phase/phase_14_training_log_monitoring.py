"""Phase 14: save, validate, and summarize the MLP training log."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .phase_1_import_library import ProjectConfig
from .table_display import style_colored_table as shared_style_colored_table


CORE_REQUIRED_COLUMNS = [
    "epoch",
    "train_loss",
    "validation_loss",
    "train_mae",
    "validation_mae",
    "validation_rmse",
    "validation_r2",
]

OPTIONAL_MONITORING_COLUMNS = [
    "train_rmse",
    "train_r2",
    "learning_rate",
    "gradient_norm",
    "epoch_time_seconds",
    "stopped_early",
]


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


def validate_training_log(training_log: pd.DataFrame) -> None:
    """Validate the minimum columns needed for monitoring and plotting."""

    if training_log is None or training_log.empty:
        raise ValueError("training_log is empty. Run Phase 13 before Phase 14.")
    missing_columns = [column for column in CORE_REQUIRED_COLUMNS if column not in training_log.columns]
    if missing_columns:
        raise ValueError(f"training_log is missing required columns: {missing_columns}")


def save_training_log(training_log: pd.DataFrame, config: ProjectConfig) -> Path:
    """Save the training log to output/logs/training_log.csv."""

    validate_training_log(training_log)
    config.logs_dir.mkdir(parents=True, exist_ok=True)
    training_log.to_csv(config.training_log_path, index=False)
    return config.training_log_path


def prepare_training_curve_data(training_log: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Prepare clean curve data from available training log columns."""

    validate_training_log(training_log)
    curve_columns = {
        "loss": ["epoch", "train_loss", "validation_loss"],
        "mae": ["epoch", "train_mae", "validation_mae"],
        "rmse": ["epoch", "train_rmse", "validation_rmse"],
        "r2": ["epoch", "train_r2", "validation_r2"],
        "learning_rate": ["epoch", "learning_rate"],
        "gradient_norm": ["epoch", "gradient_norm"],
    }
    curves: dict[str, pd.DataFrame] = {}
    for curve_name, columns in curve_columns.items():
        available_columns = [column for column in columns if column in training_log.columns]
        if "epoch" in available_columns and len(available_columns) > 1:
            curves[curve_name] = training_log[available_columns].copy()
    return curves


def build_training_log_summary(training_log: pd.DataFrame, log_path: Path) -> pd.DataFrame:
    """Build a one-row summary of the training run."""

    validate_training_log(training_log)
    best_row = training_log.loc[training_log["validation_loss"].idxmin()]
    final_row = training_log.iloc[-1]
    total_time = (
        float(training_log["epoch_time_seconds"].sum())
        if "epoch_time_seconds" in training_log.columns
        else pd.NA
    )
    return pd.DataFrame(
        [
            {
                "epochs_ran": int(training_log["epoch"].max()),
                "best_epoch": int(best_row["epoch"]),
                "best_validation_loss": float(best_row["validation_loss"]),
                "best_validation_rmse": float(best_row["validation_rmse"]),
                "best_validation_r2": float(best_row["validation_r2"]),
                "final_train_loss": float(final_row["train_loss"]),
                "final_validation_loss": float(final_row["validation_loss"]),
                "total_training_time_seconds": total_time,
                "stopped_early": bool(final_row.get("stopped_early", False)),
                "saved_log_path": str(log_path),
            }
        ]
    )


def build_training_log_column_check(training_log: pd.DataFrame) -> pd.DataFrame:
    """Show which monitoring columns are available."""

    rows = []
    for column in CORE_REQUIRED_COLUMNS:
        rows.append(
            {
                "column": column,
                "type": "required",
                "available": column in training_log.columns,
                "purpose": "Needed for core training monitoring",
            }
        )
    for column in OPTIONAL_MONITORING_COLUMNS:
        rows.append(
            {
                "column": column,
                "type": "optional",
                "available": column in training_log.columns,
                "purpose": "Useful for deeper diagnosis if available",
            }
        )
    return pd.DataFrame(rows)


def build_training_log_tail(training_log: pd.DataFrame, rows: int = 5) -> pd.DataFrame:
    """Return the last training log rows for notebook inspection."""

    preferred_columns = [
        "epoch",
        "train_loss",
        "validation_loss",
        "train_mae",
        "validation_mae",
        "validation_rmse",
        "validation_r2",
        "learning_rate",
        "gradient_norm",
        "epoch_time_seconds",
        "stopped_early",
    ]
    available_columns = [column for column in preferred_columns if column in training_log.columns]
    return training_log[available_columns].tail(rows).copy()


def build_phase_14_summary(training_log: pd.DataFrame, log_path: Path) -> dict[str, pd.DataFrame]:
    """Build display-ready Phase 14 summary tables."""

    return {
        "training_log_summary": build_training_log_summary(training_log, log_path),
        "column_check": build_training_log_column_check(training_log),
        "training_log_tail": build_training_log_tail(training_log),
    }


def display_phase_14_summary(summary: dict[str, pd.DataFrame]) -> None:
    """Display Phase 14 tables in a notebook, with a plain fallback."""

    sections = [
        ("### Training log summary", "training_log_summary", "Phase 14 training log summary"),
        ("### Column check", "column_check", "Training log column availability"),
        ("### Last epochs", "training_log_tail", "Last rows from training_log"),
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


def run_phase_14_training_log_monitoring(
    training_log: pd.DataFrame,
    config: ProjectConfig,
) -> tuple[Path, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Run Phase 14: save log, prepare curve data, and build summary tables."""

    log_path = save_training_log(training_log, config)
    curve_data = prepare_training_curve_data(training_log)
    summary = build_phase_14_summary(training_log, log_path)
    return log_path, curve_data, summary
