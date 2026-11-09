"""Paths for the DINOv3 route, built on the project's shared 2_Code/paths.py (one place for every path).

Adds only what DINOv3 needs on top of paths.py:
  DINO_RESULTS  4_Results/DINOv3/<run_name>/   results of each run
  DINO_CACHE    6_Work/dinov3_cache/<model>/   cached DINOv3 patch features (big, rebuildable)
"""
from __future__ import annotations

import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent          # 2_Code
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import paths as P  # noqa: E402

P.ensure_code_on_path()

PROJECT_ROOT = P.PROJECT_ROOT
TRAIN_VAL = P.TRAIN_VAL                                  # Train/Val dataset only. The Test_Dataset is never read here.
BASELINES = P.BASELINES
RESULTS = P.RESULTS
DINO_RESULTS = P.RESULTS / "DINOv3"
DINO_CACHE = P.WORK / "dinov3_cache"


def qa_csv() -> Path:
    """Table with the per-scene dark-water brightness used for the 'hazy' flag (same rule as the baselines)."""
    for c in (P.QA_CSV, P.RESULTS / "Data_quality" / "QA_download.csv", P.WORK / "QA_download.csv"):
        if Path(c).exists():
            return Path(c)
    raise SystemExit("QA_download.csv not found (looked in 3_Data, 4_Results/Data_quality and 6_Work). "
                     "Run the download step first: python 2_Code/run_all.py --step download")


def unet_cv_results() -> Path:
    return P.unet_dir("gpu") / "cv_results.csv"
