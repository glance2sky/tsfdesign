"""V4a: 仅开启自适应路径融合 (Adaptive Path Fusion only)"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment_v3 as experiment

# 仅开启自适应路径融合
experiment.USE_MULTISCALE_PROJECTION = False
experiment.USE_ADAPTIVE_PATH_FUSION = True
experiment.RESULT_FILENAME = "experiment_results_v4a_adaptive_only.json"

if __name__ == "__main__":
    experiment.main()
