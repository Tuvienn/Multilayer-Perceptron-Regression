"""Phase 7: scikit-learn preprocessing pipeline without data leakage."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from .table_display import style_colored_table as shared_style_colored_table
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .phase_1_import_library import ProjectConfig


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

def get_feature_groups(X_train: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical_features = [
        column
        for column in X_train.columns
        if X_train[column].dtype == "object" or str(X_train[column].dtype).startswith("category")
    ]
    if "zipcode" in X_train.columns and "zipcode" not in categorical_features:
        categorical_features.append("zipcode")
    numerical_features = [column for column in X_train.columns if column not in categorical_features]
    return numerical_features, categorical_features


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def create_preprocessing_pipeline(
    numerical_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )
    transformers = []
    if numerical_features:
        transformers.append(("numerical", numerical_pipeline, numerical_features))
    if categorical_features:
        transformers.append(("categorical", categorical_pipeline, categorical_features))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def to_dense_float32(array) -> np.ndarray:
    if hasattr(array, "toarray"):
        array = array.toarray()
    return np.asarray(array, dtype=np.float32)


def get_processed_feature_names(
    preprocessor: ColumnTransformer,
    transformed_width: int | None = None,
) -> list[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        if transformed_width is None:
            try:
                placeholder = pd.DataFrame(
                    {column: [0] for column in preprocessor.feature_names_in_}
                )
                transformed_width = preprocessor.transform(placeholder).shape[1]
            except Exception:
                transformed_width = len(getattr(preprocessor, "feature_names_in_", []))
        return [f"feature_{index}" for index in range(transformed_width)]


def build_preprocessing_steps_table() -> pd.DataFrame:
    """Describe the sklearn preprocessing pipeline used in Phase 7."""

    return pd.DataFrame(
        [
            {
                "feature_group": "Numerical",
                "step": "SimpleImputer(strategy='median')",
                "library": "sklearn",
                "purpose": "Fill missing numeric values using train median",
            },
            {
                "feature_group": "Numerical",
                "step": "StandardScaler()",
                "library": "sklearn",
                "purpose": "Scale numeric features for stable MLP training",
            },
            {
                "feature_group": "Categorical",
                "step": "SimpleImputer(strategy='most_frequent')",
                "library": "sklearn",
                "purpose": "Fill missing categorical values using train mode",
            },
            {
                "feature_group": "Categorical",
                "step": "OneHotEncoder(handle_unknown='ignore')",
                "library": "sklearn",
                "purpose": "Encode categories and avoid errors on unseen validation/test categories",
            },
            {
                "feature_group": "All",
                "step": "ColumnTransformer + Pipeline",
                "library": "sklearn",
                "purpose": "Apply separate preprocessing branches without data leakage",
            },
        ]
    )


def build_leakage_rule_table() -> pd.DataFrame:
    """Summarize fit/transform behavior for each split."""

    return pd.DataFrame(
        [
            {
                "split": "Training",
                "method": "fit_transform()",
                "meaning": "Learn imputer values, scaler statistics, and one-hot categories from train",
            },
            {
                "split": "Validation",
                "method": "transform()",
                "meaning": "Reuse preprocessing learned from train",
            },
            {
                "split": "Testing",
                "method": "transform()",
                "meaning": "Reuse preprocessing learned from train",
            },
        ]
    )


def build_feature_group_table(
    numerical_features: list[str],
    categorical_features: list[str],
) -> pd.DataFrame:
    """Show which features are routed to each preprocessing branch."""

    return pd.DataFrame(
        [
            {
                "feature_group": "Numerical",
                "feature_count": len(numerical_features),
                "meaning": "Cac cot dang so, duoc impute median va scale",
                "example_features": ", ".join(numerical_features[:8]),
            },
            {
                "feature_group": "Categorical",
                "feature_count": len(categorical_features),
                "meaning": "Cac cot dang nhom/ma danh muc, duoc impute mode va one-hot encode",
                "example_features": ", ".join(categorical_features[:8]),
            },
        ]
    )


def count_processed_features_by_group(feature_names: list[str], group_name: str) -> int:
    """Count transformed feature names produced by one ColumnTransformer branch."""

    prefix = f"{group_name}__"
    return sum(str(feature_name).startswith(prefix) for feature_name in feature_names)


def build_feature_group_count_table(
    numerical_features: list[str],
    categorical_features: list[str],
    feature_names: list[str],
) -> pd.DataFrame:
    """Count raw and processed features for each preprocessing group."""

    numerical_processed_count = count_processed_features_by_group(feature_names, "numerical")
    categorical_processed_count = count_processed_features_by_group(feature_names, "categorical")
    known_processed_count = numerical_processed_count + categorical_processed_count
    if feature_names and known_processed_count == 0:
        numerical_processed_count = len(numerical_features)
        categorical_processed_count = max(0, len(feature_names) - numerical_processed_count)
    total_raw_count = len(numerical_features) + len(categorical_features)
    total_processed_count = len(feature_names)
    return pd.DataFrame(
        [
            {
                "feature_group": "Numerical",
                "raw_feature_count": len(numerical_features),
                "processed_feature_count": numerical_processed_count,
                "change_after_preprocessing": numerical_processed_count - len(numerical_features),
                "reason": "Imputer/scaler khong tao them cot, chi bien doi gia tri",
            },
            {
                "feature_group": "Categorical",
                "raw_feature_count": len(categorical_features),
                "processed_feature_count": categorical_processed_count,
                "change_after_preprocessing": categorical_processed_count - len(categorical_features),
                "reason": "One-Hot Encoding bien moi category thanh mot cot nhi phan",
            },
            {
                "feature_group": "Total",
                "raw_feature_count": total_raw_count,
                "processed_feature_count": total_processed_count,
                "change_after_preprocessing": total_processed_count - total_raw_count,
                "reason": "Tong so cot dau vao cua MLP sau khi impute, scale va encode",
            },
        ]
    )


def build_processed_shape_table(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    X_train_processed: np.ndarray,
    X_val_processed: np.ndarray,
    X_test_processed: np.ndarray,
) -> pd.DataFrame:
    """Compare raw X shapes with transformed array shapes."""

    rows = []
    for split_name, X_raw, X_processed in [
        ("train", X_train, X_train_processed),
        ("validation", X_val, X_val_processed),
        ("test", X_test, X_test_processed),
    ]:
        rows.append(
            {
                "split": split_name,
                "raw_rows": X_raw.shape[0],
                "raw_features": X_raw.shape[1],
                "processed_rows": X_processed.shape[0],
                "processed_features": X_processed.shape[1],
                "dtype": str(X_processed.dtype),
            }
        )
    return pd.DataFrame(rows)


def build_processed_quality_table(
    X_train_processed: np.ndarray,
    X_val_processed: np.ndarray,
    X_test_processed: np.ndarray,
) -> pd.DataFrame:
    """Check that processed arrays are numeric, finite, and ready for PyTorch."""

    rows = []
    for split_name, X_processed in [
        ("train", X_train_processed),
        ("validation", X_val_processed),
        ("test", X_test_processed),
    ]:
        rows.append(
            {
                "split": split_name,
                "nan_values": int(np.isnan(X_processed).sum()),
                "infinite_values": int(np.isinf(X_processed).sum()),
                "finite_ratio": float(np.isfinite(X_processed).mean()),
                "mean": float(np.mean(X_processed)),
                "std": float(np.std(X_processed)),
            }
        )
    return pd.DataFrame(rows)


def build_artifact_table(config: ProjectConfig, feature_names: list[str]) -> pd.DataFrame:
    """Show saved preprocessing artifacts and feature-name count."""

    return pd.DataFrame(
        [
            {
                "artifact": "preprocess_pipeline.joblib",
                "path": str(config.pipeline_path),
                "exists": config.pipeline_path.exists(),
                "feature_names": len(feature_names),
            }
        ]
    )


def build_phase_7_summary(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    X_train_processed: np.ndarray,
    X_val_processed: np.ndarray,
    X_test_processed: np.ndarray,
    config: ProjectConfig,
    feature_names: list[str],
    numerical_features: list[str],
    categorical_features: list[str],
) -> dict[str, pd.DataFrame]:
    """Build colored-table friendly summaries for Phase 7."""

    return {
        "preprocessing_steps": build_preprocessing_steps_table(),
        "leakage_rules": build_leakage_rule_table(),
        "feature_groups": build_feature_group_table(numerical_features, categorical_features),
        "feature_group_counts": build_feature_group_count_table(
            numerical_features,
            categorical_features,
            feature_names,
        ),
        "processed_shapes": build_processed_shape_table(
            X_train,
            X_val,
            X_test,
            X_train_processed,
            X_val_processed,
            X_test_processed,
        ),
        "processed_quality": build_processed_quality_table(
            X_train_processed,
            X_val_processed,
            X_test_processed,
        ),
        "artifacts": build_artifact_table(config, feature_names),
    }


def display_phase_7_summary(summary: dict[str, pd.DataFrame]) -> None:
    """Display Phase 7 summary tables in a notebook, with a plain fallback."""

    sections = [
        ("### Preprocessing steps", "preprocessing_steps", "Phase 7 sklearn preprocessing steps"),
        ("### Data leakage rules", "leakage_rules", "Fit only on train, transform validation/test"),
        ("### Feature groups", "feature_groups", "Numerical and categorical feature groups"),
        ("### Feature count after preprocessing", "feature_group_counts", "Feature count by group before and after preprocessing"),
        ("### Processed shapes", "processed_shapes", "Raw X shape vs processed array shape"),
        ("### Processed quality check", "processed_quality", "Numeric array quality after preprocessing"),
        ("### Saved artifacts", "artifacts", "Preprocessing artifact saved to output"),
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


def run_phase_7_preprocessing_pipeline(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    config: ProjectConfig,
):
    """Fit preprocessing on training data, then transform validation/test."""

    numerical_features, categorical_features = get_feature_groups(X_train)
    preprocessor = create_preprocessing_pipeline(numerical_features, categorical_features)
    X_train_processed = to_dense_float32(preprocessor.fit_transform(X_train))
    X_val_processed = to_dense_float32(preprocessor.transform(X_val))
    X_test_processed = to_dense_float32(preprocessor.transform(X_test))
    config.pipeline_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, config.pipeline_path)
    feature_names = get_processed_feature_names(preprocessor, X_train_processed.shape[1])
    return (
        X_train_processed,
        X_val_processed,
        X_test_processed,
        preprocessor,
        feature_names,
        numerical_features,
        categorical_features,
    )
