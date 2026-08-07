"""V5: calibrated adaptive paths with trainable multi-scale corrections."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment_v3 as experiment


experiment.USE_MULTISCALE_PROJECTION = True
experiment.USE_ADAPTIVE_PATH_FUSION = True
experiment.USE_PATH_AMPLITUDE_CALIBRATION = True
experiment.RESULT_FILENAME = "experiment_results_v5_calibrated_multiscale.json"


if __name__ == "__main__":
    experiment.main()
