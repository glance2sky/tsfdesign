"""V4c: 同时开启多尺度时间投影和自适应路径融合"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment_v3 as experiment

# 同时开启两个模块
experiment.USE_MULTISCALE_PROJECTION = True
experiment.USE_ADAPTIVE_PATH_FUSION = True
experiment.RESULT_FILENAME = "experiment_results_v4c_both.json"

if __name__ == "__main__":
    experiment.main()
