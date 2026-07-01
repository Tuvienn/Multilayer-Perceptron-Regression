"""Phase 20: train scikit-learn comparison models."""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from .phase_1_import_library import ProjectConfig
from .phase_18_evaluate_mlp import regression_metrics
from .table_display import style_colored_table as shared_style_colored_table


DEFAULT_INFERENCE_REPEATS = 30


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


def build_baseline_models(config: ProjectConfig) -> dict[str, object]:
    """Create baseline models for fair comparison with the MLP."""

    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regression": DecisionTreeRegressor(
            random_state=config.random_state,
            max_depth=20,
            min_samples_leaf=5,
        ),
        "Random Forest Regression": RandomForestRegressor(
            n_estimators=100,
            max_depth=24,
            min_samples_leaf=2,
            random_state=config.random_state,
            n_jobs=-1,
        ),
    }


def build_baseline_rule_table() -> pd.DataFrame:
    """Explain the fair-comparison setup."""

    return pd.DataFrame(
        [
            {
                "rule": "Same split",
                "phase_20_action": "Use the same training and testing sets as MLP",
                "reason": "Avoid comparing models on different data",
            },
            {
                "rule": "Same preprocessing",
                "phase_20_action": "Use X_train_processed and X_test_processed from Phase 7",
                "reason": "Keep imputation, encoding and scaling consistent",
            },
            {
                "rule": "Same target transform",
                "phase_20_action": "Train on the same y scale as the MLP, then inverse-transform for reporting",
                "reason": "Metrics should be comparable on original price scale",
            },
            {
                "rule": "Same metrics",
                "phase_20_action": "Report MAE, MSE, RMSE, R2, training time and mean inference time",
                "reason": "Compare both accuracy and cost using repeated predict timing",
            },
        ]
    )


def validate_inference_repeats(n_repeats: int) -> int:
    repeats = int(n_repeats)
    if repeats <= 0:
        raise ValueError("n_repeats must be a positive integer for inference timing.")
    return repeats


def measure_predict_time(model, X_test, n_repeats: int = DEFAULT_INFERENCE_REPEATS) -> tuple[np.ndarray, float, float]:
    """Run model.predict repeatedly on the same test set and return prediction timing stats."""

    repeats = validate_inference_repeats(n_repeats)
    timings: list[float] = []
    y_pred: np.ndarray | None = None
    for _ in range(repeats):
        inference_start = time.perf_counter()
        current_pred = model.predict(X_test)
        timings.append(time.perf_counter() - inference_start)
        if y_pred is None:
            y_pred = np.asarray(current_pred)

    timing_array = np.asarray(timings, dtype=float)
    inference_time_mean = float(timing_array.mean())
    inference_time_std = float(timing_array.std(ddof=1)) if repeats > 1 else 0.0
    if y_pred is None:
        raise RuntimeError("No predictions were produced during baseline inference timing.")
    return y_pred, inference_time_mean, inference_time_std


def build_model_description_table() -> pd.DataFrame:
    """Describe why each baseline is included."""

    return pd.DataFrame(
        [
            {
                "model": "Linear Regression",
                "type": "linear baseline",
                "purpose": "Check whether simple linear relationships are enough",
            },
            {
                "model": "Decision Tree Regression",
                "type": "single nonlinear tree",
                "purpose": "Capture nonlinear rules but may overfit",
            },
            {
                "model": "Random Forest Regression",
                "type": "ensemble tree baseline",
                "purpose": "Stronger nonlinear baseline with reduced variance",
            },
        ]
    )


def build_phase_20_summary(comparison_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build display-ready Phase 20 summary tables."""

    ranked_df = comparison_df.sort_values("RMSE").reset_index(drop=True).copy()
    ranked_df.insert(0, "rmse_rank", range(1, len(ranked_df) + 1))
    return {
        "baseline_results": ranked_df,
        "baseline_rules": build_baseline_rule_table(),
        "model_descriptions": build_model_description_table(),
    }


def display_phase_20_summary(summary: dict[str, pd.DataFrame]) -> None:
    """Display Phase 20 summary tables in a notebook, with a plain fallback."""

    sections = [
        ("### Baseline model results", "baseline_results", "Phase 20 baseline metrics on test set"),
        ("### Fair comparison rules", "baseline_rules", "How Phase 20 keeps comparison fair"),
        ("### Model descriptions", "model_descriptions", "Why these baseline models are included"),
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


def run_phase_20_comparison_models(
    X_train,
    y_train,
    X_test,
    y_test,
    config: ProjectConfig,
    inverse_transform: Callable | None = None,
    n_repeats: int = DEFAULT_INFERENCE_REPEATS,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, pd.DataFrame]]:
    """Train baseline regressors and evaluate them on the test set."""

    y_train_array = np.asarray(y_train, dtype=float).reshape(-1)
    y_test_array = np.asarray(y_test, dtype=float).reshape(-1)
    models = build_baseline_models(config)

    rows: list[dict] = []
    fitted_models: dict[str, object] = {}
    for model_name, model in models.items():
        train_start = time.perf_counter()
        model.fit(X_train, y_train_array)
        training_time = time.perf_counter() - train_start

        y_pred, inference_time_mean, inference_time_std = measure_predict_time(
            model,
            X_test,
            n_repeats=n_repeats,
        )

        metric_y_true = y_test_array
        metric_y_pred = y_pred
        if inverse_transform is not None:
            metric_y_true = inverse_transform(metric_y_true)
            metric_y_pred = inverse_transform(metric_y_pred)

        rows.append(
            {
                "model": model_name,
                **regression_metrics(metric_y_true, metric_y_pred),
                "training_time_seconds": training_time,
                "inference_time_seconds": inference_time_mean,
                "inference_time_mean_seconds": inference_time_mean,
                "inference_time_std_seconds": inference_time_std,
                "inference_time_repeats": int(n_repeats),
                "target_scale": "original price scale after inverse transform" if inverse_transform is not None else "model target scale",
            }
        )
        fitted_models[model_name] = model

    comparison_df = pd.DataFrame(rows)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(config.results_dir / "baseline_model_metrics.csv", index=False)
    summary = build_phase_20_summary(comparison_df)
    return comparison_df, fitted_models, summary
