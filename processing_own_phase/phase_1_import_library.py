"""Phase 1: imports, paths, output folders, and reproducibility setup."""

from __future__ import annotations

import json
import os
import random
import sys
from importlib import util as importlib_util
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp") / "matplotlib-cache"))

PROJECT_LIBRARY_CHECKS = (
    ("NumPy", "numpy", "Array math, random seed, numeric processing"),
    ("Pandas", "pandas", "DataFrame loading, EDA, tabular transforms"),
    ("Matplotlib", "matplotlib", "EDA, training, prediction plots"),
    ("Scikit-learn", "sklearn", "Splitting, preprocessing, metrics, baselines"),
    ("PyTorch", "torch", "MLP model, Dataset/DataLoader, training loop"),
    ("Joblib", "joblib", "Save preprocessing pipeline artifacts"),
)


def ensure_project_root_on_path(root_dir: Path = ROOT_DIR) -> Path:
    """Allow notebook cells to import phase modules after running Phase 1 with %run."""

    root_dir = Path(root_dir).resolve()
    root_path = str(root_dir)
    if root_path not in sys.path:
        sys.path.insert(0, root_path)
    return root_dir


@dataclass
class ProjectConfig:
    """Central configuration used by every phase."""

    root_dir: Path = ROOT_DIR
    data_path: Path = ROOT_DIR / "multilayer_perceptron_regression" / "data" / "kc_house_data.csv"
    output_dir: Path = ROOT_DIR / "output"
    artifacts_dir: Path = ROOT_DIR / "output" / "artifacts"
    logs_dir: Path = ROOT_DIR / "output" / "logs"
    results_dir: Path = ROOT_DIR / "output" / "results"
    plots_dir: Path = ROOT_DIR / "output" / "plots"
    random_state: int = 42
    train_ratio: float = 0.80
    validation_ratio: float = 0.10
    test_ratio: float = 0.10
    target_column: str = "price"
    batch_size: int = 128
    epochs: int = 150
    learning_rate: float = 1e-3
    hidden_units: tuple[int, ...] = (128, 64)
    dropout: float = 0.10
    weight_decay: float = 1e-5
    loss_function: str = "mse"
    huber_delta: float = 1.0
    gradient_clip_norm: float | None = None
    patience: int = 15
    min_delta: float = 1e-4
    log_transform_target: bool = True

    def validate(self) -> None:
        """Validate configuration values before the pipeline uses them."""

        split_total = self.train_ratio + self.validation_ratio + self.test_ratio
        if abs(split_total - 1.0) > 1e-8:
            raise ValueError("train_ratio + validation_ratio + test_ratio must equal 1.0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.loss_function not in {"mse", "huber", "smooth_l1"}:
            raise ValueError("loss_function must be one of: mse, huber, smooth_l1")
        if self.huber_delta <= 0:
            raise ValueError("huber_delta must be positive")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive when provided")

    @property
    def eda_plots_dir(self) -> Path:
        return self.plots_dir / "eda"

    @property
    def training_plots_dir(self) -> Path:
        return self.plots_dir / "training"

    @property
    def prediction_plots_dir(self) -> Path:
        return self.plots_dir / "predictions"

    @property
    def comparison_plots_dir(self) -> Path:
        return self.plots_dir / "comparison"

    @property
    def training_log_path(self) -> Path:
        return self.logs_dir / "training_log.csv"

    @property
    def best_model_path(self) -> Path:
        return self.artifacts_dir / "best_mlp_model.pth"

    @property
    def pipeline_path(self) -> Path:
        return self.artifacts_dir / "preprocess_pipeline.joblib"

    @property
    def config_path(self) -> Path:
        return self.artifacts_dir / "config.json"

    @property
    def mlp_metrics_path(self) -> Path:
        return self.results_dir / "mlp_metrics.csv"

    @property
    def predictions_path(self) -> Path:
        return self.results_dir / "mlp_predictions.csv"

    @property
    def comparison_path(self) -> Path:
        return self.results_dir / "model_comparison.csv"

    @property
    def hyperparameter_results_path(self) -> Path:
        return self.results_dir / "hyperparameter_results.csv"

    def ensure_output_directories(self) -> list[Path]:
        """Create all expected output folders."""

        self.validate()
        directories = [
            self.output_dir,
            self.artifacts_dir,
            self.logs_dir,
            self.results_dir,
            self.plots_dir,
            self.eda_plots_dir,
            self.training_plots_dir,
            self.prediction_plots_dir,
            self.comparison_plots_dir,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        return directories

    def to_serializable_dict(self) -> dict[str, Any]:
        values = asdict(self)
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in values.items()
        }


def set_random_seed(seed: int = 42) -> dict[str, bool | int]:
    """Set deterministic seeds for Python, NumPy, and PyTorch if installed."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    numpy_seed_set = False
    try:
        import numpy as np

        np.random.seed(seed)
        numpy_seed_set = True
    except ModuleNotFoundError:
        numpy_seed_set = False
    torch_seed_set = False
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        torch_seed_set = True
    except ModuleNotFoundError:
        torch_seed_set = False
    return {
        "seed": seed,
        "numpy_seed_set": numpy_seed_set,
        "torch_seed_set": torch_seed_set,
    }


def check_library_status() -> dict[str, dict[str, str | bool]]:
    """Check whether core project libraries are available in the active kernel."""

    status = {}
    for display_name, module_name, purpose in PROJECT_LIBRARY_CHECKS:
        available = importlib_util.find_spec(module_name) is not None
        status[display_name] = {
            "module": module_name,
            "purpose": purpose,
            "available": available,
        }
    return status


def save_config(config: ProjectConfig) -> Path:
    """Persist the project config as JSON for reproducibility."""

    config.config_path.parent.mkdir(parents=True, exist_ok=True)
    config.config_path.write_text(
        json.dumps(config.to_serializable_dict(), indent=2),
        encoding="utf-8",
    )
    return config.config_path


def initialize_project(config: ProjectConfig | None = None) -> ProjectConfig:
    """Create output folders, set seeds, and persist config."""

    ensure_project_root_on_path()
    config = config or ProjectConfig()
    config.ensure_output_directories()
    set_random_seed(config.random_state)
    save_config(config)
    return config


def normalize_path_collection(path_collection) -> dict[str, Path]:
    """Normalize dict/list/single path collections into a named path dictionary."""

    if isinstance(path_collection, dict):
        return {
            str(name): Path(path)
            for name, path in path_collection.items()
        }
    if isinstance(path_collection, (str, Path)):
        path = Path(path_collection)
        return {path.stem: path}

    normalized_paths = {}
    for index, path in enumerate(path_collection, start=1):
        path = Path(path)
        normalized_paths[path.stem or f"path_{index}"] = path
    return normalized_paths


def display_path_collection(path_collection) -> None:
    """Display generated paths compactly inside a notebook."""

    normalized_paths = normalize_path_collection(path_collection)

    try:
        from IPython.display import Image, display
    except ModuleNotFoundError:
        for name, path in normalized_paths.items():
            print(name, path)
        return

    for name, path in normalized_paths.items():
        print(name, path)
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"} and path.exists():
            display(Image(filename=str(path)))


def build_phase_1_html(config: ProjectConfig) -> str:
    """Build a visual Phase 1 overview for notebook display."""

    path_rows = [
        ("Data", config.data_path),
        ("Output", config.output_dir),
        ("Artifacts", config.artifacts_dir),
        ("Logs", config.logs_dir),
        ("Results", config.results_dir),
        ("Plots", config.plots_dir),
    ]
    config_items = [
        ("Random state", config.random_state),
        ("Split ratio", f"{config.train_ratio:.0%} / {config.validation_ratio:.0%} / {config.test_ratio:.0%}"),
        ("Target", config.target_column),
        ("Batch size", config.batch_size),
        ("Epochs", config.epochs),
        ("Learning rate", config.learning_rate),
        ("Hidden units", config.hidden_units),
        ("Dropout", config.dropout),
        ("Patience", config.patience),
    ]
    library_status = check_library_status()
    ready_count = sum(1 for item in library_status.values() if item["available"])
    total_count = len(library_status)
    library_cards = "\n".join(
        f"""
        <div style="border:1px solid {'#b7e4c7' if item['available'] else '#fecaca'};border-radius:8px;background:{'#f0fdf4' if item['available'] else '#fff1f2'};padding:10px 12px;">
          <div style="display:flex;align-items:center;justify-content:space-between;gap:10px;">
            <strong style="color:#111827;font-size:14px;">{escape(name)}</strong>
            <span style="border-radius:999px;padding:3px 8px;font-size:11px;font-weight:700;color:{'#166534' if item['available'] else '#991b1b'};background:{'#dcfce7' if item['available'] else '#fee2e2'};">
              {'READY' if item['available'] else 'MISSING'}
            </span>
          </div>
          <div style="font-family:Menlo,Consolas,monospace;font-size:12px;color:#4b5563;margin-top:5px;">import {escape(str(item['module']))}</div>
          <div style="font-size:12px;color:#4b5563;margin-top:6px;line-height:1.35;">{escape(str(item['purpose']))}</div>
        </div>
        """
        for name, item in library_status.items()
    )
    path_cards = "\n".join(
        f"""
        <div style="padding:10px 12px;border:1px solid #d7dee8;border-radius:8px;background:#ffffff;">
          <div style="display:flex;justify-content:space-between;align-items:center;gap:10px;">
            <div style="font-size:12px;color:#526070;text-transform:uppercase;font-weight:700;">{escape(label)}</div>
            <span style="font-size:11px;color:{'#166534' if Path(path).exists() else '#92400e'};background:{'#dcfce7' if Path(path).exists() else '#fef3c7'};border-radius:999px;padding:2px 7px;">
              {'exists' if Path(path).exists() else 'pending'}
            </span>
          </div>
          <div style="font-family:Menlo,Consolas,monospace;font-size:12px;color:#18202a;word-break:break-all;margin-top:6px;">{escape(str(path))}</div>
        </div>
        """
        for label, path in path_rows
    )
    config_cards = "\n".join(
        f"""
        <div style="display:flex;justify-content:space-between;gap:12px;padding:8px 0;border-bottom:1px solid #e8edf3;">
          <span style="color:#526070;">{escape(label)}</span>
          <strong style="color:#18202a;">{escape(str(value))}</strong>
        </div>
        """
        for label, value in config_items
    )
    return f"""
    <div style="border:1px solid #cfd8e3;border-radius:8px;overflow:hidden;background:#f8fafc;margin:8px 0 16px;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
      <div style="background:#12343b;color:white;padding:16px 18px;">
        <div style="display:flex;justify-content:space-between;gap:12px;align-items:flex-start;flex-wrap:wrap;">
          <div>
            <div style="font-size:20px;font-weight:800;letter-spacing:0;">Phase 1 - Project Setup</div>
            <div style="font-size:13px;opacity:0.9;margin-top:4px;">Import libraries, configure paths, set random seed, prepare output folders</div>
          </div>
          <div style="background:rgba(255,255,255,0.14);border:1px solid rgba(255,255,255,0.25);border-radius:8px;padding:8px 10px;min-width:150px;">
            <div style="font-size:11px;opacity:0.78;text-transform:uppercase;font-weight:700;">Library status</div>
            <div style="font-size:22px;font-weight:800;line-height:1.1;margin-top:2px;">{ready_count}/{total_count} ready</div>
          </div>
        </div>
      </div>
      <div style="padding:16px 18px;">
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin-bottom:14px;">
          <div style="background:#ffffff;border:1px solid #d7dee8;border-radius:8px;padding:12px;">
            <div style="font-size:12px;color:#64748b;font-weight:700;text-transform:uppercase;">Seed</div>
            <div style="font-size:22px;font-weight:800;color:#111827;margin-top:2px;">{config.random_state}</div>
          </div>
          <div style="background:#ffffff;border:1px solid #d7dee8;border-radius:8px;padding:12px;">
            <div style="font-size:12px;color:#64748b;font-weight:700;text-transform:uppercase;">Split</div>
            <div style="font-size:22px;font-weight:800;color:#111827;margin-top:2px;">{config.train_ratio:.0%}/{config.validation_ratio:.0%}/{config.test_ratio:.0%}</div>
          </div>
          <div style="background:#ffffff;border:1px solid #d7dee8;border-radius:8px;padding:12px;">
            <div style="font-size:12px;color:#64748b;font-weight:700;text-transform:uppercase;">Target</div>
            <div style="font-size:22px;font-weight:800;color:#111827;margin-top:2px;">{escape(config.target_column)}</div>
          </div>
          <div style="background:#ffffff;border:1px solid #d7dee8;border-radius:8px;padding:12px;">
            <div style="font-size:12px;color:#64748b;font-weight:700;text-transform:uppercase;">Framework</div>
            <div style="font-size:22px;font-weight:800;color:#111827;margin-top:2px;">PyTorch</div>
          </div>
        </div>

        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px;margin-bottom:14px;">
          {library_cards}
        </div>

        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px;">
          <div>
            <div style="font-size:13px;color:#334155;font-weight:800;margin:0 0 8px;">Project paths</div>
            <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:10px;">
              {path_cards}
            </div>
          </div>
          <div>
            <div style="font-size:13px;color:#334155;font-weight:800;margin:0 0 8px;">Main configuration</div>
            <div style="background:#ffffff;border:1px solid #d7dee8;border-radius:8px;padding:4px 14px;">
              {config_cards}
            </div>
          </div>
        </div>

        <div style="margin-top:14px;display:flex;gap:8px;flex-wrap:wrap;color:#526070;font-size:13px;">
          <span style="background:#fee2e2;color:#991b1b;border-radius:999px;padding:5px 10px;font-weight:700;">TensorFlow/Keras not used</span>
          <span style="background:#e0f2fe;color:#075985;border-radius:999px;padding:5px 10px;font-weight:700;">NumPy/Pandas/Matplotlib/Scikit-learn allowed</span>
          <span style="background:#dcfce7;color:#166534;border-radius:999px;padding:5px 10px;font-weight:700;">PyTorch MLP required</span>
        </div>
      </div>
    </div>
    """


def display_phase_1_overview(config: ProjectConfig) -> None:
    """Display Phase 1 setup as HTML in notebooks, with a plain fallback."""

    html = build_phase_1_html(config)
    try:
        from IPython.display import HTML, display
    except ModuleNotFoundError:
        print("Phase 1 - Project Setup")
        print(json.dumps(config.to_serializable_dict(), indent=2))
        print("Library status:")
        for name, item in check_library_status().items():
            state = "READY" if item["available"] else "MISSING"
            print(f"- {name}: {state} ({item['module']})")
        return
    display(HTML(html))


def run_phase_1_import_library(config: ProjectConfig | None = None, show_html: bool = True) -> ProjectConfig:
    """Initialize Phase 1 and optionally display a visual notebook summary."""

    config = initialize_project(config)
    if show_html:
        display_phase_1_overview(config)
    return config


DEFAULT_CONFIG = ProjectConfig()


if __name__ == "__main__":
    config = run_phase_1_import_library()
