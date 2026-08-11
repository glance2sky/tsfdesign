"""V9a: v7c plus multi-scale hyperbolic patch and variable tokens."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment_v3 as experiment


experiment.USE_MULTISCALE_PROJECTION = False
experiment.USE_ADAPTIVE_PATH_FUSION = True
experiment.USE_PATH_AMPLITUDE_CALIBRATION = True
experiment.USE_OUTPUT_MULTISCALE_RESIDUAL = False
experiment.USE_FREQUENCY_RESIDUAL = False
experiment.USE_TREND_DIFFERENCE_RESIDUAL = True
experiment.USE_EXPLICIT_PERIODIC_RESIDUAL = True
experiment.USE_VARIABLE_HIERARCHY = False
experiment.USE_TEMPORAL_HIERARCHY = False
experiment.USE_RECURSIVE_TEMPORAL_HIERARCHY = False
experiment.USE_PATCH_TOKENS = True
experiment.PATCH_LENGTHS = (8, 16, 32)
experiment.PATCH_STRIDES = (4, 8, 16)
experiment.PATCH_HIDDEN_DIM = 64
experiment.RESULT_FILENAME = "experiment_results_v9a_patch_tokens.json"


if __name__ == "__main__":
    experiment.main()
