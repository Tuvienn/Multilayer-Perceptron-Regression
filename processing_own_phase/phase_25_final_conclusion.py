"""Phase 25: generate the final project conclusion."""

from __future__ import annotations

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


def _fmt(value, precision: int = 3) -> str:
    """Format values for conclusion text."""

    if value is None or pd.isna(value):
        return "N/A"
    if isinstance(value, str):
        return value
    return f"{float(value):,.{precision}f}"


def build_key_result_table(model_comparison_df: pd.DataFrame, analysis: dict[str, object]) -> pd.DataFrame:
    """Build key final results."""

    return pd.DataFrame(
        [
            {
                "item": "Best MAE model",
                "value": analysis.get("best_mae_model", "N/A"),
                "evidence": f"MAE = {_fmt(analysis.get('best_mae_value'))}",
            },
            {
                "item": "Best RMSE model",
                "value": analysis.get("best_rmse_model", "N/A"),
                "evidence": f"RMSE = {_fmt(analysis.get('best_rmse_value'))}",
            },
            {
                "item": "Best R2 model",
                "value": analysis.get("best_r2_model", "N/A"),
                "evidence": f"R2 = {_fmt(analysis.get('best_r2_value'))}",
            },
            {
                "item": "Recommended model",
                "value": analysis.get("recommended_model", "N/A"),
                "evidence": analysis.get("recommendation_reason", "See Phase 22 analysis."),
            },
            {
                "item": "Models compared",
                "value": len(model_comparison_df),
                "evidence": ", ".join(model_comparison_df["model"].astype(str).tolist()),
            },
        ]
    )


def build_project_summary_table() -> pd.DataFrame:
    """Summarize the completed project workflow."""

    return pd.DataFrame(
        [
            {"stage": "Data split", "summary": "Raw data split into train/validation/test with 80/10/10 ratio"},
            {"stage": "Preprocessing", "summary": "Missing imputation, encoding and scaling handled with sklearn pipeline"},
            {"stage": "Modeling", "summary": "MLP Regression implemented with PyTorch"},
            {"stage": "Monitoring", "summary": "Training log, validation metrics, gradient norm and early stopping tracked"},
            {"stage": "Evaluation", "summary": "MLP evaluated on test set and compared with sklearn baselines"},
            {"stage": "Artifacts", "summary": "Model, pipeline, metrics, predictions, plots and config saved to output/"},
        ]
    )


def build_limitations_table() -> pd.DataFrame:
    """List important limitations for an honest final conclusion."""

    return pd.DataFrame(
        [
            {
                "limitation": "MLP interpretability",
                "note": "MLP is harder to explain directly than tree or linear baselines.",
            },
            {
                "limitation": "Hyperparameter search",
                "note": "Phase 17 uses a controlled small grid unless expanded manually.",
            },
            {
                "limitation": "Feature importance",
                "note": "Random Forest feature importance is a reference, not the exact MLP decision logic.",
            },
            {
                "limitation": "Final metric reliability",
                "note": "Conclusion depends on running enough epochs and using the best restored model.",
            },
        ]
    )


def build_next_steps_table() -> pd.DataFrame:
    """Suggest practical next steps."""

    return pd.DataFrame(
        [
            {"next_step": "Run full notebook", "reason": "Confirm all outputs are generated in the same kernel"},
            {"next_step": "Tune hyperparameters", "reason": "Improve MLP performance beyond the default config"},
            {"next_step": "Inspect high-error cases", "reason": "Understand where the model struggles most"},
            {"next_step": "Try more features", "reason": "Location and interaction features may improve price prediction"},
            {"next_step": "Compare deployment tradeoffs", "reason": "Balance accuracy, inference speed and interpretability"},
        ]
    )


def generate_final_conclusion(model_comparison_df: pd.DataFrame, analysis: dict[str, object]) -> str:
    """Generate a concise final conclusion paragraph."""

    best_rmse_model = analysis.get("best_rmse_model", "N/A")
    best_r2_model = analysis.get("best_r2_model", "N/A")
    recommended_model = analysis.get("recommended_model", best_rmse_model)
    mlp_vs_baseline = analysis.get(
        "mlp_vs_baseline",
        "MLP should only be considered better if the metrics support that conclusion.",
    )
    overfitting_status = analysis.get("overfitting_status", "unknown")
    return (
        "Dự án đã xây dựng pipeline dự đoán giá nhà bằng MLP Regression với PyTorch, "
        "trong đó dữ liệu được chia 80/10/10 và toàn bộ preprocessing được fit trên training set để tránh data leakage. "
        f"Sau khi đánh giá trên test set, model có RMSE tốt nhất là {best_rmse_model}, "
        f"model có R2 tốt nhất là {best_r2_model}. "
        f"Nhận xét MLP so với baseline: {mlp_vs_baseline} "
        f"Tín hiệu overfitting hiện tại: {overfitting_status}. "
        f"Model được đề xuất cho kết quả hiện tại là {recommended_model}, "
        "nhưng lựa chọn cuối nên cân bằng giữa accuracy, inference speed và interpretability."
    )


def build_phase_25_summary(
    model_comparison_df: pd.DataFrame,
    analysis: dict[str, object],
) -> dict[str, pd.DataFrame]:
    """Build display-ready Phase 25 summary tables."""

    return {
        "project_summary": build_project_summary_table(),
        "key_results": build_key_result_table(model_comparison_df, analysis),
        "limitations": build_limitations_table(),
        "next_steps": build_next_steps_table(),
    }


def save_final_conclusion(final_conclusion: str, config: ProjectConfig) -> Path:
    """Save final conclusion as Markdown."""

    output_path = config.results_dir / "final_conclusion.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(final_conclusion + "\n", encoding="utf-8")
    return output_path


def display_phase_25_summary(summary: dict[str, pd.DataFrame], final_conclusion: str) -> None:
    """Display final conclusion and tables in a notebook, with a plain fallback."""

    sections = [
        ("### Project summary", "project_summary", "Completed project workflow"),
        ("### Key results", "key_results", "Final key results"),
        ("### Limitations", "limitations", "Important limitations"),
        ("### Next steps", "next_steps", "Suggested next steps"),
    ]
    try:
        from IPython.display import Markdown, display
    except ModuleNotFoundError:
        print(final_conclusion)
        for title, key, _caption in sections:
            print(title.replace("#", "").strip())
            print(summary[key])
        return

    display(Markdown("### Final conclusion"))
    display(Markdown(final_conclusion))
    for title, key, caption in sections:
        display(Markdown(title))
        display(style_colored_table(summary[key], caption))


def run_phase_25_final_conclusion(
    model_comparison_df: pd.DataFrame,
    analysis: dict[str, object],
    config: ProjectConfig,
) -> tuple[str, dict[str, pd.DataFrame], dict[str, Path]]:
    """Run Phase 25 final conclusion generation."""

    final_conclusion = generate_final_conclusion(model_comparison_df, analysis)
    summary = build_phase_25_summary(model_comparison_df, analysis)
    output_paths = {"final_conclusion": save_final_conclusion(final_conclusion, config)}
    return final_conclusion, summary, output_paths
