"""V6b: v4d plus frequency-domain periodic residual."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment_v3 as experiment


experiment.USE_MULTISCALE_PROJECTION = False
experiment.USE_ADAPTIVE_PATH_FUSION = True
experiment.USE_PATH_AMPLITUDE_CALIBRATION = True
experiment.USE_OUTPUT_MULTISCALE_RESIDUAL = False
experiment.USE_FREQUENCY_RESIDUAL = True
experiment.RESULT_FILENAME = "experiment_results_v6b_frequency.json"


if __name__ == "__main__":
    experiment.main()
