"""V4b: 仅开启多尺度时间投影 (Multi-Scale Temporal Projection only)"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment_v3 as experiment

# 仅开启多尺度时间投影
experiment.USE_MULTISCALE_PROJECTION = True
experiment.USE_ADAPTIVE_PATH_FUSION = False
experiment.RESULT_FILENAME = "experiment_results_v4b_multiscale_only.json"

if __name__ == "__main__":
    experiment.main()
