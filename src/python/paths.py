from pathlib import Path
import os

def find_project_root(marker=".git"):
    """Search upwards from the current file or CWD for a directory containing the marker."""
    try:
        base = Path(__file__).resolve()
    except NameError:
        # __file__ is not defined in notebooks; use CWD
        base = Path.cwd().resolve()
    
    for parent in [base] + list(base.parents):
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Could not find project root with marker '{marker}'")

project_root = find_project_root()
data_path = project_root / "data"
training_path = data_path / "training"
training_path.mkdir(parents=True, exist_ok=True)
validation_path = data_path / "validation"
validation_path.mkdir(parents=True, exist_ok=True)
models_path = data_path / "models"
models_path.mkdir(parents=True, exist_ok=True)