"""Phase 22: analyze model comparison results and choose a practical model."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .phase_1_import_library import ProjectConfig
from .table_display import style_colored_table as shared_style_colored_table


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


def _best_row(model_comparison_df: pd.DataFrame, metric: str, ascending: bool = True) -> dict[str, object]:
    """Return the best row for one metric."""

    metric_df = model_comparison_df.dropna(subset=[metric])
    if metric_df.empty:
        return {"model": "N/A", metric: None}
    return metric_df.sort_values(metric, ascending=ascending).iloc[0].to_dict()


def _format_number(value, precision: int = 3) -> str:
    """Format a value for readable analysis text."""

    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):,.{precision}f}"


def analyze_overfitting(training_log: pd.DataFrame | None = None) -> dict[str, object]:
    """Analyze overfitting from train/validation loss."""

    if training_log is None or training_log.empty:
        return {
            "status": "unknown",
            "final_train_loss": None,
            "final_validation_loss": None,
            "final_loss_gap": None,
            "best_epoch": None,
            "note": "Training log not provided.",
        }

    required_columns = ["epoch", "train_loss", "validation_loss"]
    missing_columns = [column for column in required_columns if column not in training_log.columns]
    if missing_columns:
        return {
            "status": "unknown",
            "final_train_loss": None,
            "final_validation_loss": None,
            "final_loss_gap": None,
            "best_epoch": None,
            "note": f"Training log is missing columns: {missing_columns}",
        }

    final_row = training_log.iloc[-1]
    best_row = training_log.loc[training_log["validation_loss"].idxmin()]
    final_train_loss = float(final_row["train_loss"])
    final_validation_loss = float(final_row["validation_loss"])
    gap = final_validation_loss - final_train_loss
    relative_gap = gap / abs(final_train_loss) if final_train_loss != 0 else 0.0

    if relative_gap > 0.20:
        status = "potential overfitting"
        note = "Validation loss is much higher than train loss."
    elif gap > 0:
        status = "small generalization gap"
        note = "Validation loss is higher than train loss, but the gap is not large."
    else:
        status = "no obvious overfitting"
        note = "Final validation loss is not higher than train loss."

    return {
        "status": status,
        "final_train_loss": final_train_loss,
        "final_validation_loss": final_validation_loss,
        "final_loss_gap": gap,
        "relative_gap": relative_gap,
        "best_epoch": int(best_row["epoch"]),
        "best_validation_loss": float(best_row["validation_loss"]),
        "note": note,
    }


def compare_mlp_with_baselines(model_comparison_df: pd.DataFrame) -> str:
    """Return a short conclusion about MLP versus baselines."""

    best_rmse = _best_row(model_comparison_df, "RMSE", ascending=True)
    best_r2 = _best_row(model_comparison_df, "R2", ascending=False)
    mlp_rows = model_comparison_df[model_comparison_df["model"].str.contains("MLP", case=False, na=False)]
    if mlp_rows.empty:
        return "MLP result is not available in the comparison table."
    mlp_model = mlp_rows.iloc[0]["model"]
    if best_rmse["model"] == mlp_model and best_r2["model"] == mlp_model:
        return "MLP is strongest on both RMSE and R2 among the compared models."
    if best_rmse["model"] == mlp_model:
        return "MLP has the best RMSE, but another model may have better R2."
    return f"MLP is not the best by RMSE; {best_rmse['model']} has lower RMSE on the test set."


def choose_recommended_model(model_comparison_df: pd.DataFrame) -> dict[str, object]:
    """Choose a practical final model from comparison metrics."""

    best_rmse = _best_row(model_comparison_df, "RMSE", ascending=True)
    best_r2 = _best_row(model_comparison_df, "R2", ascending=False)
    recommended_model = best_rmse["model"]
    reason = (
        f"{recommended_model} has the lowest RMSE, so it makes fewer large errors on the test set. "
        "Final selection should still consider inference time and interpretability."
    )
    if best_rmse["model"] != best_r2["model"]:
        reason += f" Note: {best_r2['model']} has the highest R2."
    return {
        "recommended_model": recommended_model,
        "primary_reason": reason,
        "best_rmse": best_rmse.get("RMSE"),
        "best_r2_model": best_r2.get("model"),
        "best_r2": best_r2.get("R2"),
    }


def analyze_results(model_comparison_df: pd.DataFrame, training_log: pd.DataFrame | None = None) -> dict[str, object]:
    """Return compact analysis for later phases such as the final conclusion."""

    if model_comparison_df is None or model_comparison_df.empty:
        raise ValueError("model_comparison_df is empty. Run Phase 21 before Phase 22.")
    best_mae = _best_row(model_comparison_df, "MAE", ascending=True)
    best_rmse = _best_row(model_comparison_df, "RMSE", ascending=True)
    best_r2 = _best_row(model_comparison_df, "R2", ascending=False)
    overfitting = analyze_overfitting(training_log)
    recommendation = choose_recommended_model(model_comparison_df)
    return {
        "best_mae_model": best_mae["model"],
        "best_mae_value": best_mae.get("MAE"),
        "best_rmse_model": best_rmse["model"],
        "best_rmse_value": best_rmse.get("RMSE"),
        "best_r2_model": best_r2["model"],
        "best_r2_value": best_r2.get("R2"),
        "mlp_vs_baseline": compare_mlp_with_baselines(model_comparison_df),
        "overfitting_note": overfitting["note"],
        "overfitting_status": overfitting["status"],
        "recommended_model": recommendation["recommended_model"],
        "recommendation_reason": recommendation["primary_reason"],
        "interpretability_note": "Linear/Tree/Random Forest baselines are easier to interpret than MLP.",
    }


def build_question_answer_table(
    model_comparison_df: pd.DataFrame,
    training_log: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Answer the Phase 22 project-plan questions."""

    analysis = analyze_results(model_comparison_df, training_log)
    fastest_train = _best_row(model_comparison_df, "training_time_seconds", ascending=True)
    fastest_inference = _best_row(model_comparison_df, "inference_time_seconds", ascending=True)
    return pd.DataFrame(
        [
            {
                "question": "Which model has the lowest MAE?",
                "answer": analysis["best_mae_model"],
                "evidence": f"MAE = {_format_number(analysis['best_mae_value'])}",
            },
            {
                "question": "Which model has the lowest RMSE?",
                "answer": analysis["best_rmse_model"],
                "evidence": f"RMSE = {_format_number(analysis['best_rmse_value'])}",
            },
            {
                "question": "Which model has the highest R2?",
                "answer": analysis["best_r2_model"],
                "evidence": f"R2 = {_format_number(analysis['best_r2_value'])}",
            },
            {
                "question": "Is MLP better than the baselines?",
                "answer": analysis["mlp_vs_baseline"],
                "evidence": "Compare MLP rank against Linear Regression, Decision Tree and Random Forest.",
            },
            {
                "question": "Is there an overfitting signal?",
                "answer": analysis["overfitting_status"],
                "evidence": analysis["overfitting_note"],
            },
            {
                "question": "Which model trains fastest?",
                "answer": fastest_train["model"],
                "evidence": f"Training time = {_format_number(fastest_train.get('training_time_seconds'), 6)} seconds",
            },
            {
                "question": "Which model predicts fastest?",
                "answer": fastest_inference["model"],
                "evidence": f"Inference time = {_format_number(fastest_inference.get('inference_time_seconds'), 6)} seconds",
            },
            {
                "question": "Which model is most suitable overall?",
                "answer": analysis["recommended_model"],
                "evidence": analysis["recommendation_reason"],
            },
        ]
    )


def build_overfitting_table(training_log: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build a display-ready overfitting analysis table."""

    overfitting = analyze_overfitting(training_log)
    return pd.DataFrame([overfitting])


def build_recommendation_table(model_comparison_df: pd.DataFrame) -> pd.DataFrame:
    """Build a final recommendation table."""

    recommendation = choose_recommended_model(model_comparison_df)
    return pd.DataFrame([recommendation])


def build_metric_ranking_table(model_comparison_df: pd.DataFrame) -> pd.DataFrame:
    """Build a sorted metric ranking table for Phase 22."""

    ranking_df = model_comparison_df.copy()
    ranking_df["mae_rank"] = ranking_df["MAE"].rank(method="min", ascending=True)
    ranking_df["rmse_rank"] = ranking_df["RMSE"].rank(method="min", ascending=True)
    ranking_df["r2_rank"] = ranking_df["R2"].rank(method="min", ascending=False)
    ranking_df["accuracy_score"] = ranking_df[["mae_rank", "rmse_rank", "r2_rank"]].sum(axis=1)
    return ranking_df.sort_values(["accuracy_score", "RMSE"]).reset_index(drop=True)


def build_phase_22_summary(
    model_comparison_df: pd.DataFrame,
    training_log: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Build display-ready Phase 22 summary tables."""

    return {
        "question_answers": build_question_answer_table(model_comparison_df, training_log),
        "metric_ranking": build_metric_ranking_table(model_comparison_df),
        "overfitting_analysis": build_overfitting_table(training_log),
        "recommendation": build_recommendation_table(model_comparison_df),
    }


def save_phase_22_outputs(
    analysis: dict[str, object],
    summary: dict[str, pd.DataFrame],
    config: ProjectConfig,
) -> dict[str, Path]:
    """Save Phase 22 analysis outputs."""

    config.results_dir.mkdir(parents=True, exist_ok=True)
    json_path = config.results_dir / "analysis_summary.json"
    csv_path = config.results_dir / "analysis_question_answers.csv"
    json_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    summary["question_answers"].to_csv(csv_path, index=False)
    return {"analysis_summary": json_path, "analysis_question_answers": csv_path}


def display_phase_22_summary(summary: dict[str, pd.DataFrame]) -> None:
    """Display Phase 22 summary tables in a notebook, with a plain fallback."""

    sections = [
        ("### Phase 22 question answers", "question_answers", "Answers to the Phase 22 analysis questions"),
        ("### Metric ranking", "metric_ranking", "Ranking models by MAE, RMSE and R2"),
        ("### Overfitting analysis", "overfitting_analysis", "Train/validation loss gap analysis"),
        ("### Final recommendation", "recommendation", "Recommended model from Phase 22"),
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
        gradient_columns = [column for column in ["MAE", "RMSE", "R2", "accuracy_score"] if column in summary[key].columns]
        display(style_colored_table(summary[key], caption, gradient_columns=gradient_columns or None))


def run_phase_22_analyze_results(
    model_comparison_df: pd.DataFrame,
    training_log: pd.DataFrame | None,
    config: ProjectConfig,
) -> tuple[dict[str, object], dict[str, pd.DataFrame], dict[str, Path]]:
    """Run Phase 22 analysis and save output artifacts."""

    analysis = analyze_results(model_comparison_df, training_log)
    summary = build_phase_22_summary(model_comparison_df, training_log)
    output_paths = save_phase_22_outputs(analysis, summary, config)
    return analysis, summary, output_paths
