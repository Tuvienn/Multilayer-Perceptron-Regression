"""Phase 17: controlled hyperparameter exploration for the PyTorch MLP."""

from __future__ import annotations

import os
from pathlib import Path
import time

os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp") / "matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .phase_1_import_library import ProjectConfig
from .table_display import style_colored_table as shared_style_colored_table


PLOT_STYLE = {
    "best": "#dc2626",
    "other": "#2563eb",
    "grid": "#cbd5e1",
}
DEFAULT_PREDICTION_GUARDRAIL_MULTIPLIER = 3.0


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


def build_hyperparameter_grid(max_trials: int | None = None) -> list[dict]:
    """Return a small, reviewable search grid with robust-loss trials.

    The curated grid keeps the notebook practical while still testing smaller
    learning rates, different hidden-layer shapes, stronger regularization and
    robust losses for expensive/high-error houses.
    """

    grid = [
        {
            "learning_rate": 0.001,
            "hidden_units": (128, 64),
            "batch_size": 128,
            "dropout": 0.10,
            "weight_decay": 1e-5,
            "l1_lambda": 1e-6,
            "loss_function": "mse",
            "huber_delta": 1.0,
            "gradient_clip_norm": 5.0,
        },
        {
            "learning_rate": 0.0005,
            "hidden_units": (128, 64),
            "batch_size": 128,
            "dropout": 0.10,
            "weight_decay": 1e-5,
            "l1_lambda": 1e-6,
            "loss_function": "huber",
            "huber_delta": 1.0,
            "gradient_clip_norm": 5.0,
        },
        {
            "learning_rate": 0.0005,
            "hidden_units": (256, 128, 64),
            "batch_size": 128,
            "dropout": 0.15,
            "weight_decay": 1e-5,
            "l1_lambda": 1e-6,
            "loss_function": "huber",
            "huber_delta": 0.75,
            "gradient_clip_norm": 5.0,
        },
        {
            "learning_rate": 0.0002,
            "hidden_units": (256, 128, 64),
            "batch_size": 64,
            "dropout": 0.20,
            "weight_decay": 1e-4,
            "l1_lambda": 1e-6,
            "loss_function": "huber",
            "huber_delta": 0.50,
            "gradient_clip_norm": 5.0,
        },
        {
            "learning_rate": 0.0002,
            "hidden_units": (128, 128),
            "batch_size": 64,
            "dropout": 0.10,
            "weight_decay": 1e-4,
            "l1_lambda": 1e-6,
            "loss_function": "smooth_l1",
            "huber_delta": 0.75,
            "gradient_clip_norm": 5.0,
        },
        {
            "learning_rate": 0.0001,
            "hidden_units": (256, 128),
            "batch_size": 128,
            "dropout": 0.20,
            "weight_decay": 1e-4,
            "l1_lambda": 1e-6,
            "loss_function": "smooth_l1",
            "huber_delta": 0.50,
            "gradient_clip_norm": 5.0,
        },
        {
            "learning_rate": 0.0005,
            "hidden_units": (64,),
            "batch_size": 64,
            "dropout": 0.00,
            "weight_decay": 0.0,
            "l1_lambda": 1e-6,
            "loss_function": "mse",
            "huber_delta": 1.0,
            "gradient_clip_norm": 5.0,
        },
        {
            "learning_rate": 0.0002,
            "hidden_units": (512, 256, 128),
            "batch_size": 128,
            "dropout": 0.20,
            "weight_decay": 1e-4,
            "l1_lambda": 1e-6,
            "loss_function": "huber",
            "huber_delta": 0.75,
            "gradient_clip_norm": 5.0,
        },
    ]
    if max_trials is not None:
        return grid[:max_trials]
    return grid


def save_hyperparameter_results(results: list[dict] | pd.DataFrame, config: ProjectConfig) -> Path:
    """Save hyperparameter plan or experiment results to CSV."""

    config.results_dir.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(config.hyperparameter_results_path, index=False)
    return config.hyperparameter_results_path


def _save_plot(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_validation_rmse_by_trial(results_df: pd.DataFrame, config: ProjectConfig) -> Path | None:
    """Plot validation RMSE on original price scale for completed trials."""

    if results_df.empty or "validation_original_rmse" not in results_df.columns:
        return None
    plot_df = results_df.dropna(subset=["validation_original_rmse"]).copy()
    if "status" in plot_df.columns:
        plot_df = plot_df[plot_df["status"].eq("completed")].copy()
    plot_df = plot_df[np.isfinite(plot_df["validation_original_rmse"].astype(float))].copy()
    if plot_df.empty:
        return None
    plot_df = plot_df.sort_values("validation_original_rmse").reset_index(drop=True)
    labels = [
        f"T{int(row.trial)}\n{row.loss_function}\n{row.hidden_units}"
        for row in plot_df.itertuples(index=False)
    ]
    colors = [PLOT_STYLE["best"]] + [PLOT_STYLE["other"]] * (len(plot_df) - 1)
    plt.figure(figsize=(11, 5.8))
    plt.bar(labels, plot_df["validation_original_rmse"], color=colors)
    plt.title("Phase 17 Hyperparameter Search - Validation RMSE")
    plt.xlabel("Trial / loss / hidden units")
    plt.ylabel("Validation RMSE on original price scale")
    plt.xticks(rotation=20, ha="right")
    plt.grid(True, axis="y", color=PLOT_STYLE["grid"], alpha=0.45)
    return _save_plot(config.training_plots_dir / "phase_17_hyperparameter_validation_rmse.png")


def create_hyperparameter_plots(results_df: pd.DataFrame, config: ProjectConfig) -> dict[str, Path]:
    """Create Phase 17 visualization outputs."""

    plot_paths: dict[str, Path] = {}
    rmse_plot = plot_validation_rmse_by_trial(results_df, config)
    if rmse_plot is not None:
        plot_paths["validation_rmse_by_trial"] = rmse_plot
    return plot_paths


def create_pending_hyperparameter_table(config: ProjectConfig, max_trials: int | None = 16) -> pd.DataFrame:
    """Create a grid table that can be filled after running experiments."""

    grid = build_hyperparameter_grid(max_trials=max_trials)
    table = pd.DataFrame(grid)
    table.insert(0, "trial", range(1, len(table) + 1))
    table["hidden_units"] = table["hidden_units"].astype(str)
    table["validation_loss"] = pd.NA
    table["validation_rmse"] = pd.NA
    table["validation_original_rmse"] = pd.NA
    table["validation_original_mae"] = pd.NA
    table["validation_original_r2"] = pd.NA
    table["min_prediction"] = pd.NA
    table["max_prediction"] = pd.NA
    table["max_actual"] = pd.NA
    table["max_reasonable_prediction"] = pd.NA
    table["non_finite_prediction_count"] = pd.NA
    table["negative_prediction_count"] = pd.NA
    table["extreme_high_prediction_count"] = pd.NA
    table["unstable_reason"] = "-"
    table["best_epoch"] = pd.NA
    table["training_time_seconds"] = pd.NA
    table["status"] = "pending"
    save_hyperparameter_results(table, config)
    return table


def build_trial_decision_summary(hyperparameter_table: pd.DataFrame) -> pd.DataFrame:
    """Build one presentation table that combines ranking and stability decisions."""

    if hyperparameter_table.empty:
        return pd.DataFrame(
            [
                {
                    "trial": "-",
                    "decision": "No trials",
                    "status": "empty",
                    "note": "No hyperparameter trials available",
                }
            ]
        )

    table = hyperparameter_table.copy()
    if "status" in table.columns:
        table["status"] = table["status"].astype(str)
    else:
        table["status"] = "pending"
    if "trial" not in table.columns:
        table.insert(0, "trial", range(1, len(table) + 1))

    metric_column = "validation_original_rmse" if "validation_original_rmse" in table.columns else "validation_loss"
    if metric_column in table.columns:
        table["_ranking_metric"] = pd.to_numeric(table[metric_column], errors="coerce")
    else:
        table["_ranking_metric"] = np.nan

    completed_mask = table["status"].eq("completed") & table["_ranking_metric"].notna()
    completed_ranked = table.loc[completed_mask, ["trial", "_ranking_metric"]].sort_values("_ranking_metric")
    rank_by_trial = {
        trial: rank
        for rank, trial in enumerate(completed_ranked["trial"].tolist(), start=1)
    }

    rows = []
    for _, row in table.iterrows():
        trial = row["trial"]
        status = str(row.get("status", "pending"))
        rank = rank_by_trial.get(trial, "-")
        unstable_reason = str(row.get("unstable_reason", "-"))
        if status == "completed" and rank == 1:
            decision = "Best candidate"
            note = "RMSE thap nhat trong cac trial completed"
            display_group = 0
        elif status == "completed":
            decision = "Valid candidate"
            note = "Trial hop le, co the so sanh voi best candidate"
            display_group = 1
        elif "unstable" in status.lower():
            decision = "Rejected: unstable prediction"
            note = unstable_reason if unstable_reason and unstable_reason != "-" else "Du doan bi guardrail danh dau bat on"
            display_group = 2
        elif status == "pending":
            decision = "Pending"
            note = "Moi la ke hoach, chua co ket qua train"
            display_group = 3
        else:
            decision = "Review"
            note = "Can xem lai status cua trial"
            display_group = 4

        rows.append(
            {
                "trial": trial,
                "decision": decision,
                "status": status,
                "learning_rate": row.get("learning_rate", pd.NA),
                "hidden_units": row.get("hidden_units", pd.NA),
                "batch_size": row.get("batch_size", pd.NA),
                "dropout": row.get("dropout", pd.NA),
                "weight_decay": row.get("weight_decay", pd.NA),
                "loss_function": row.get("loss_function", pd.NA),
                "validation_original_rmse": row.get("validation_original_rmse", pd.NA),
                "validation_original_mae": row.get("validation_original_mae", pd.NA),
                "validation_original_r2": row.get("validation_original_r2", pd.NA),
                "best_epoch": row.get("best_epoch", pd.NA),
                "training_time_seconds": row.get("training_time_seconds", pd.NA),
                "unstable_reason": unstable_reason,
                "note": note,
                "_display_group": display_group,
                "_rank_sort": rank if isinstance(rank, int) else 999,
            }
        )

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(["_display_group", "_rank_sort", "trial"]).reset_index(drop=True)
    summary = summary.drop(columns=["_display_group", "_rank_sort"])
    numeric_display_columns = [
        "validation_original_rmse",
        "validation_original_mae",
        "validation_original_r2",
        "best_epoch",
        "training_time_seconds",
    ]
    for column in numeric_display_columns:
        if column in summary.columns:
            summary[column] = pd.to_numeric(summary[column], errors="coerce")
    return summary


def _phase_17_trial_row_style(row: pd.Series) -> list[str]:
    """Return row-level CSS for the combined Phase 17 trial summary table."""

    decision = str(row.get("decision", "")).lower()
    if "best candidate" in decision:
        style = "background-color: #dcfce7; color: #14532d; font-weight: 800;"
    elif "valid candidate" in decision:
        style = "background-color: #eff6ff; color: #1e3a8a; font-weight: 700;"
    elif "unstable" in decision:
        style = "background-color: #fff1f2; color: #991b1b; font-weight: 700;"
    elif "pending" in decision:
        style = "background-color: #f3f4f6; color: #374151;"
    else:
        style = "background-color: #fff7ed; color: #9a3412;"
    return [style for _ in row]


def style_phase_17_trial_summary(df: pd.DataFrame, caption: str):
    """Style the combined Phase 17 table so ranking and rejected trials are easy to see."""

    metric_columns = [
        column
        for column in ["validation_original_rmse", "validation_original_mae", "validation_original_r2"]
        if column in df.columns and pd.to_numeric(df[column], errors="coerce").notna().any()
    ]
    emphasized_columns = [column for column in ["decision", "status", "note"] if column in df.columns]
    return (
        style_colored_table(
            df,
            caption=caption,
            precision=3,
            gradient_columns=metric_columns,
        )
        .apply(_phase_17_trial_row_style, axis=1)
        .set_properties(
            subset=emphasized_columns,
            **{"font-weight": "800"},
        )
    )


def build_phase_17_summary(
    hyperparameter_table: pd.DataFrame,
    results_path: Path,
    ran_experiments: bool,
    plot_paths: dict[str, Path] | None = None,
) -> dict[str, pd.DataFrame]:
    """Build display-ready Phase 17 summary tables."""

    overview = pd.DataFrame(
        [
            {
                "total_trials_in_table": len(hyperparameter_table),
                "ran_experiments": ran_experiments,
                "results_path": str(results_path),
                "note": "One table combines completed ranking, unstable prediction flags, and presentation notes",
            }
        ]
    )
    parameter_meaning = pd.DataFrame(
        [
            {"hyperparameter": "learning_rate", "meaning": "Adam step size; too high may oscillate"},
            {"hyperparameter": "hidden_units", "meaning": "Number of neurons/layers in the MLP"},
            {"hyperparameter": "batch_size", "meaning": "Samples per weight update"},
            {"hyperparameter": "dropout", "meaning": "Regularization to reduce overfitting"},
            {"hyperparameter": "weight_decay", "meaning": "L2 regularization strength in Adam"},
            {"hyperparameter": "l1_lambda", "meaning": "L1 regularization strength added to the training loss"},
            {"hyperparameter": "loss_function", "meaning": "mse, huber, or smooth_l1 objective"},
            {"hyperparameter": "huber_delta", "meaning": "Transition point for robust losses on log target scale"},
            {"hyperparameter": "gradient_clip_norm", "meaning": "Optional max gradient norm used to stabilize trials"},
            {"hyperparameter": "prediction_guardrail_multiplier", "meaning": "Flags trials whose original-scale predictions are implausibly high"},
        ]
    )
    completed_df = hyperparameter_table[hyperparameter_table.get("status", "").eq("completed")].copy()
    if not completed_df.empty and "validation_original_rmse" in completed_df.columns:
        ranked_trials = completed_df.sort_values("validation_original_rmse").reset_index(drop=True)
    elif not completed_df.empty:
        ranked_trials = completed_df.sort_values("validation_loss").reset_index(drop=True)
    else:
        ranked_trials = pd.DataFrame([{"trial": "-", "status": "No completed trials yet"}])
    if "status" in hyperparameter_table.columns:
        unstable_trials = hyperparameter_table[
            hyperparameter_table["status"].astype(str).str.contains("unstable", case=False, na=False)
        ].copy()
    else:
        unstable_trials = pd.DataFrame()
    if unstable_trials.empty:
        unstable_trials = pd.DataFrame([{"trial": "-", "status": "No unstable prediction trials"}])
    trial_summary = build_trial_decision_summary(hyperparameter_table)
    summary = {
        "phase_17_overview": overview,
        "trial_summary": trial_summary,
        "hyperparameter_table": hyperparameter_table,
        "ranked_trials": ranked_trials,
        "unstable_trials": unstable_trials,
        "created_plots": pd.DataFrame(
            [{"plot_name": name, "saved_path": str(path), "status": "created"} for name, path in (plot_paths or {}).items()]
        )
        if plot_paths
        else pd.DataFrame([{"plot_name": "-", "saved_path": "-", "status": "No Phase 17 plots created"}]),
        "parameter_meaning": parameter_meaning,
    }
    return summary


def display_phase_17_summary(summary: dict[str, pd.DataFrame]) -> None:
    """Display Phase 17 summary tables in a notebook, with a plain fallback."""

    if "trial_summary" not in summary and "hyperparameter_table" in summary:
        summary = {
            **summary,
            "trial_summary": build_trial_decision_summary(summary["hyperparameter_table"]),
        }

    sections = [
        ("### Hyperparameter exploration overview", "phase_17_overview", "Phase 17 exploration overview"),
        ("### Hyperparameter trial summary", "trial_summary", "One table: ranking, decisions, unstable flags and notes"),
        ("### Created hyperparameter plots", "created_plots", "Phase 17 saved plots"),
        ("### Parameter meaning", "parameter_meaning", "What each hyperparameter controls"),
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
        if key == "trial_summary":
            display(style_phase_17_trial_summary(summary[key], caption))
        else:
            display(style_colored_table(summary[key], caption))


def run_phase_17_hyperparameter_exploration_plan(
    config: ProjectConfig,
    max_trials: int | None = 16,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Create the Phase 17 pending experiment table without training models."""

    table = create_pending_hyperparameter_table(config=config, max_trials=max_trials)
    summary = build_phase_17_summary(
        hyperparameter_table=table,
        results_path=config.hyperparameter_results_path,
        ran_experiments=False,
    )
    return table, summary


def summarize_prediction_guardrails(
    y_true,
    y_pred,
    prediction_guardrail_multiplier: float = DEFAULT_PREDICTION_GUARDRAIL_MULTIPLIER,
    target_scale: str = "model target scale",
) -> dict[str, float | int | bool | str]:
    """Flag hyperparameter trials with implausible original-price predictions."""

    y_true_array = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred_array = np.asarray(y_pred, dtype=float).reshape(-1)
    finite_true = y_true_array[np.isfinite(y_true_array)]
    finite_pred = y_pred_array[np.isfinite(y_pred_array)]
    max_actual = float(finite_true.max()) if finite_true.size else float("nan")
    min_prediction = float(finite_pred.min()) if finite_pred.size else float("nan")
    max_prediction = float(finite_pred.max()) if finite_pred.size else float("nan")
    max_reasonable_prediction = max_actual * prediction_guardrail_multiplier if np.isfinite(max_actual) else float("nan")
    non_finite_prediction_count = int((~np.isfinite(y_pred_array)).sum())
    negative_prediction_count = int((finite_pred < 0).sum()) if finite_pred.size else 0
    if np.isfinite(max_reasonable_prediction) and finite_pred.size:
        extreme_high_prediction_count = int((finite_pred > max_reasonable_prediction).sum())
    else:
        extreme_high_prediction_count = 0

    reasons = []
    if non_finite_prediction_count:
        reasons.append(f"{non_finite_prediction_count} non-finite predictions")
    if "original price scale" in target_scale and negative_prediction_count:
        reasons.append(f"{negative_prediction_count} negative price predictions")
    if "original price scale" in target_scale and extreme_high_prediction_count:
        reasons.append(
            f"{extreme_high_prediction_count} predictions above {prediction_guardrail_multiplier:g}x validation max price"
        )
    is_unstable = bool(reasons)
    return {
        "min_prediction": min_prediction,
        "max_prediction": max_prediction,
        "max_actual": max_actual,
        "max_reasonable_prediction": max_reasonable_prediction,
        "non_finite_prediction_count": non_finite_prediction_count,
        "negative_prediction_count": negative_prediction_count,
        "extreme_high_prediction_count": extreme_high_prediction_count,
        "is_unstable": is_unstable,
        "unstable_reason": "; ".join(reasons) if reasons else "-",
    }


def safe_regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Calculate metrics, returning a clearly bad score for non-finite predictions."""

    y_true_array = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred_array = np.asarray(y_pred, dtype=float).reshape(-1)
    finite_mask = np.isfinite(y_true_array) & np.isfinite(y_pred_array)
    if not finite_mask.any():
        return {"MAE": float("inf"), "MSE": float("inf"), "RMSE": float("inf"), "R2": float("nan")}
    y_true_array = y_true_array[finite_mask]
    y_pred_array = y_pred_array[finite_mask]
    residuals = y_true_array - y_pred_array
    mae = float(np.mean(np.abs(residuals)))
    mse = float(np.mean(residuals**2))
    rmse = float(np.sqrt(mse))
    denominator = float(np.sum((y_true_array - np.mean(y_true_array)) ** 2))
    r2 = float(1 - np.sum(residuals**2) / denominator) if denominator > 0 else float("nan")
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


def run_hyperparameter_exploration(
    X_train_processed,
    y_train,
    X_val_processed,
    y_val,
    input_dim: int,
    config: ProjectConfig,
    max_epochs_per_trial: int = 20,
    max_trials: int | None = 8,
    inverse_transform=None,
    patience_per_trial: int | None = None,
) -> pd.DataFrame:
    """Train a small grid of MLP configurations and save validation results."""

    from .phase_9_pytorch_dataset import create_tensor_dataset
    from .phase_10_pytorch_dataloader import create_data_loader
    from .phase_11_build_mlp_model import build_mlp_model
    from .phase_12_model_configuration import ModelHyperparameters
    from .phase_13_train_mlp_pytorch import train_mlp_pytorch
    from .phase_16_early_stopping import EarlyStopping
    from .phase_18_evaluate_mlp import predict_mlp

    train_dataset = create_tensor_dataset(X_train_processed, y_train)
    val_dataset = create_tensor_dataset(X_val_processed, y_val)
    rows: list[dict] = []
    trial_patience = patience_per_trial or min(config.patience, max(5, max_epochs_per_trial // 4))
    prediction_guardrail_multiplier = float(
        getattr(config, "prediction_guardrail_multiplier", DEFAULT_PREDICTION_GUARDRAIL_MULTIPLIER)
    )
    search_artifact_dir = config.artifacts_dir / "hyperparameter_search"
    search_artifact_dir.mkdir(parents=True, exist_ok=True)

    for trial_index, trial_config in enumerate(build_hyperparameter_grid(max_trials=max_trials), start=1):
        start_time = time.perf_counter()
        train_loader = create_data_loader(
            train_dataset,
            batch_size=trial_config["batch_size"],
            shuffle=True,
            random_state=config.random_state + trial_index,
        )
        val_loader = create_data_loader(val_dataset, batch_size=trial_config["batch_size"], shuffle=False)
        model = build_mlp_model(
            input_dim=input_dim,
            hidden_units=trial_config["hidden_units"],
            dropout=trial_config["dropout"],
        )
        hyperparameters = ModelHyperparameters(
            learning_rate=trial_config["learning_rate"],
            batch_size=trial_config["batch_size"],
            epochs=max_epochs_per_trial,
            hidden_units=trial_config["hidden_units"],
            dropout=trial_config["dropout"],
            weight_decay=trial_config["weight_decay"],
            l1_lambda=trial_config["l1_lambda"],
            patience=trial_patience,
            min_delta=config.min_delta,
            loss_function=trial_config["loss_function"],
            huber_delta=trial_config["huber_delta"],
            gradient_clip_norm=trial_config.get("gradient_clip_norm"),
        )
        early_stopping = EarlyStopping(
            patience=trial_patience,
            min_delta=config.min_delta,
            model_path=search_artifact_dir / f"trial_{trial_index:02d}_best.pth",
            restore_best_model=True,
        )
        history = train_mlp_pytorch(model, train_loader, val_loader, hyperparameters, early_stopping=early_stopping)
        best_row = history.sort_values("validation_loss").iloc[0]
        validation_y_true, validation_y_pred, validation_inference_time, device = predict_mlp(model, val_loader)
        metric_y_true = validation_y_true
        metric_y_pred = validation_y_pred
        target_scale = "model target scale"
        if inverse_transform is not None:
            metric_y_true = inverse_transform(validation_y_true)
            metric_y_pred = inverse_transform(validation_y_pred)
            target_scale = "original price scale after inverse transform"
        prediction_guardrails = summarize_prediction_guardrails(
            metric_y_true,
            metric_y_pred,
            prediction_guardrail_multiplier=prediction_guardrail_multiplier,
            target_scale=target_scale,
        )
        validation_metrics = safe_regression_metrics(metric_y_true, metric_y_pred)
        status = "unstable_prediction" if prediction_guardrails["is_unstable"] else "completed"
        elapsed_time = time.perf_counter() - start_time
        rows.append(
            {
                "trial": trial_index,
                "learning_rate": trial_config["learning_rate"],
                "hidden_units": str(trial_config["hidden_units"]),
                "batch_size": trial_config["batch_size"],
                "dropout": trial_config["dropout"],
                "weight_decay": trial_config["weight_decay"],
                "l1_lambda": trial_config["l1_lambda"],
                "loss_function": trial_config["loss_function"],
                "huber_delta": trial_config["huber_delta"],
                "gradient_clip_norm": trial_config.get("gradient_clip_norm"),
                "best_epoch": int(best_row["epoch"]),
                "validation_loss": float(best_row["validation_loss"]),
                "validation_rmse": float(best_row["validation_rmse"]),
                "validation_original_mae": float(validation_metrics["MAE"]),
                "validation_original_mse": float(validation_metrics["MSE"]),
                "validation_original_rmse": float(validation_metrics["RMSE"]),
                "validation_original_r2": float(validation_metrics["R2"]),
                "min_prediction": float(prediction_guardrails["min_prediction"]),
                "max_prediction": float(prediction_guardrails["max_prediction"]),
                "max_actual": float(prediction_guardrails["max_actual"]),
                "max_reasonable_prediction": float(prediction_guardrails["max_reasonable_prediction"]),
                "non_finite_prediction_count": int(prediction_guardrails["non_finite_prediction_count"]),
                "negative_prediction_count": int(prediction_guardrails["negative_prediction_count"]),
                "extreme_high_prediction_count": int(prediction_guardrails["extreme_high_prediction_count"]),
                "unstable_reason": str(prediction_guardrails["unstable_reason"]),
                "training_time_seconds": float(elapsed_time),
                "validation_inference_time_seconds": float(validation_inference_time),
                "target_scale": target_scale,
                "device": device,
                "best_model_path": str(early_stopping.model_path),
                "status": status,
            }
        )

    results_df = pd.DataFrame(rows)
    save_hyperparameter_results(results_df, config)
    return results_df


def run_phase_17_hyperparameter_exploration(
    X_train_processed,
    y_train,
    X_val_processed,
    y_val,
    input_dim: int,
    config: ProjectConfig,
    max_epochs_per_trial: int = 20,
    max_trials: int | None = 8,
    inverse_transform=None,
    patience_per_trial: int | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Run a limited Phase 17 grid search and return display-ready summaries."""

    results_df = run_hyperparameter_exploration(
        X_train_processed=X_train_processed,
        y_train=y_train,
        X_val_processed=X_val_processed,
        y_val=y_val,
        input_dim=input_dim,
        config=config,
        max_epochs_per_trial=max_epochs_per_trial,
        max_trials=max_trials,
        inverse_transform=inverse_transform,
        patience_per_trial=patience_per_trial,
    )
    plot_paths = create_hyperparameter_plots(results_df, config)
    summary = build_phase_17_summary(
        hyperparameter_table=results_df,
        results_path=config.hyperparameter_results_path,
        ran_experiments=True,
        plot_paths=plot_paths,
    )
    return results_df, summary
