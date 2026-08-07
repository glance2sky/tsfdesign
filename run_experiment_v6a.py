"""V6a: v4d plus output-space multi-scale residual."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment_v3 as experiment


experiment.USE_MULTISCALE_PROJECTION = False
experiment.USE_ADAPTIVE_PATH_FUSION = True
experiment.USE_PATH_AMPLITUDE_CALIBRATION = True
experiment.USE_OUTPUT_MULTISCALE_RESIDUAL = True
experiment.USE_FREQUENCY_RESIDUAL = False
experiment.RESULT_FILENAME = "experiment_results_v6a_output_multiscale.json"


if __name__ == "__main__":
    experiment.main()
