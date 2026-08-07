"""V4d: adaptive path fusion + path amplitude calibration (no multiscale)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment_v3 as experiment

experiment.USE_MULTISCALE_PROJECTION = False
experiment.USE_ADAPTIVE_PATH_FUSION = True
experiment.USE_PATH_AMPLITUDE_CALIBRATION = True
experiment.RESULT_FILENAME = "experiment_results_v4d_adaptive_calibration.json"

if __name__ == "__main__":
    experiment.main()
