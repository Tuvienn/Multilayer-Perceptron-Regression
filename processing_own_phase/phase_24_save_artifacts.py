"""Phase 24: save final project artifacts and create an artifact manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .phase_1_import_library import ProjectConfig
from .table_display import style_colored_table as shared_style_colored_table


try:
    import torch
except ModuleNotFoundError:
    torch = None


def require_torch_artifacts():
    """Return torch or raise a helpful setup error."""

    if torch is None:
        raise ModuleNotFoundError(
            "PyTorch is required to save the MLP model state_dict. "
            "Use the Jupyter kernel that has torch installed, then rerun Phase 24."
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


def save_model(model, config: ProjectConfig) -> Path:
    """Save the current MLP model weights."""

    torch_module = require_torch_artifacts()
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    torch_module.save(model.state_dict(), config.best_model_path)
    return config.best_model_path


def save_pipeline(preprocessor, config: ProjectConfig) -> Path:
    """Save the sklearn preprocessing pipeline."""

    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, config.pipeline_path)
    return config.pipeline_path


def save_dataframe(df: pd.DataFrame, path: str | Path) -> Path:
    """Save a DataFrame as CSV."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def save_config(config: ProjectConfig) -> Path:
    """Save the project config as JSON."""

    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    config.config_path.write_text(json.dumps(config.to_serializable_dict(), indent=2), encoding="utf-8")
    return config.config_path


def build_artifact_manifest(artifact_paths: dict[str, Path]) -> pd.DataFrame:
    """Build a table describing saved artifacts."""

    purpose_map = {
        "best_mlp_model": "Best/restored PyTorch MLP model weights",
        "preprocess_pipeline": "Fitted sklearn preprocessing pipeline",
        "config": "Project configuration and hyperparameters",
        "model_comparison": "Metrics for MLP and baseline models",
        "mlp_predictions": "Actual and predicted house prices for the test set",
        "training_log": "Epoch-level PyTorch training log",
        "analysis_summary": "Phase 22 final analysis summary",
        "plots_dir": "All generated plots grouped by phase",
    }
    rows = []
    for name, path in artifact_paths.items():
        path = Path(path)
        rows.append(
            {
                "artifact": name,
                "path": str(path),
                "exists": path.exists(),
                "purpose": purpose_map.get(name, "Project output artifact"),
            }
        )
    return pd.DataFrame(rows)


def save_artifact_manifest(manifest_df: pd.DataFrame, config: ProjectConfig) -> Path:
    """Save artifact manifest as CSV."""

    manifest_path = config.results_dir / "artifact_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(manifest_path, index=False)
    return manifest_path


def build_phase_24_summary(manifest_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build display-ready Phase 24 summary tables."""

    missing_df = manifest_df[~manifest_df["exists"]].copy()
    if missing_df.empty:
        missing_df = pd.DataFrame([{"artifact": "-", "path": "-", "exists": True, "purpose": "All artifacts exist"}])
    return {
        "artifact_manifest": manifest_df,
        "missing_artifacts": missing_df,
    }


def display_phase_24_summary(summary: dict[str, pd.DataFrame]) -> None:
    """Display Phase 24 summary tables in a notebook, with a plain fallback."""

    sections = [
        ("### Artifact manifest", "artifact_manifest", "Saved project artifacts"),
        ("### Missing artifact check", "missing_artifacts", "Artifacts that still need attention"),
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


def run_phase_24_save_artifacts(
    model,
    preprocessor,
    model_comparison_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    config: ProjectConfig,
    analysis: dict[str, Any] | None = None,
) -> tuple[dict[str, Path], dict[str, pd.DataFrame]]:
    """Run Phase 24 artifact saving."""

    artifact_paths = {
        "best_mlp_model": save_model(model, config),
        "preprocess_pipeline": save_pipeline(preprocessor, config),
        "config": save_config(config),
        "model_comparison": save_dataframe(model_comparison_df, config.comparison_path),
        "mlp_predictions": save_dataframe(predictions_df, config.predictions_path),
        "training_log": config.training_log_path,
        "analysis_summary": config.results_dir / "analysis_summary.json",
        "plots_dir": config.plots_dir,
    }
    if analysis is not None:
        artifact_paths["analysis_summary"].write_text(json.dumps(analysis, indent=2), encoding="utf-8")

    manifest_df = build_artifact_manifest(artifact_paths)
    manifest_path = save_artifact_manifest(manifest_df, config)
    artifact_paths["artifact_manifest"] = manifest_path
    summary = build_phase_24_summary(build_artifact_manifest(artifact_paths))
    return artifact_paths, summary
