"""ETTh1 experiment entry point for the error-driven HyperbolicTSF head.

This keeps the v3 training protocol and enables only the two new forecasting
head components:
  - multi-scale temporal projection;
  - variable- and horizon-conditioned path fusion.
"""

from __future__ import annotations

import run_experiment_v3 as experiment


experiment.USE_MULTISCALE_PROJECTION = True
experiment.USE_ADAPTIVE_PATH_FUSION = True
experiment.RESULT_FILENAME = "experiment_results_v4.json"


if __name__ == "__main__":
    experiment.main()
