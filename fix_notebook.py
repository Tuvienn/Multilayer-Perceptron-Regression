import json

file_path = "/Users/vientu/MACHINE LEARNING/Multilayer Perceptron Regression/multilayer_perceptron_regression/evaluation/mlp_regression.ipynb"
with open(file_path, "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        cell["source"] = [
            "import sys\n",
            "from pathlib import Path\n",
            "\n",
            "sys.path.append(\"/Users/vientu/MACHINE LEARNING/Multilayer Perceptron Regression\")\n",
            "from processing_own_phase.phase_1_import_library import run_phase_1_import_library\n",
            "\n",
            "config = run_phase_1_import_library()\n"
        ]
        break

with open(file_path, "w") as f:
    json.dump(nb, f, indent=1)
