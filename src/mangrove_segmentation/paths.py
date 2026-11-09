"""Every folder and file the pipeline uses, defined in one place.

PROJECT_ROOT is the SDP folder (the one that holds 2_Code, 3_Data and 4_Results). It is:
  * the environment variable MANGROVE_ROOT, if it is set, otherwise
  * the parent of the folder this file is in (2_Code).

Layout
  3_Data/      small inputs that are kept: sites, labels, the image list, folds, QA table
  4_Results/   tables, figures and reports (Baselines, Unet, Unet_gpu, ...)
  6_Work/      everything big that the pipeline rebuilds: raw images, chips, datasets, land cover
Nothing here is specific to Windows, Linux or Colab: only pathlib is used.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent

# Windows: never crash on a non-ASCII character (e.g. '±') when the output goes to a notebook or a file
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except Exception:  # Jupyter's output stream has no reconfigure()
        pass
PROJECT_ROOT = Path(os.environ["MANGROVE_ROOT"]).expanduser().resolve() if os.environ.get("MANGROVE_ROOT") \
    else CODE.parent

# ---- small inputs (3_Data) ----
DATA = PROJECT_ROOT / "3_Data"
SITES = DATA / "sites" / "sites.geojson"
LABELS = DATA / "labels"
SCENES = DATA / "scenes_selected.csv"            # the exact Sentinel-2 images (site, item_id)
SCENES_DETAIL = DATA / "scenes_selected_detail.csv"
FOLDS = DATA / "folds_site_cv.csv"               # site -> fold / role (cv, test_fixed, external_fixed)
QA_CSV = DATA / "QA_download.csv"                # per site/date water brightness -> 'hazy' flag
if not QA_CSV.exists() and (PROJECT_ROOT / "4_Results" / "Data_quality" / "QA_download.csv").exists():
    QA_CSV = PROJECT_ROOT / "4_Results" / "Data_quality" / "QA_download.csv"
REFERENCE_CATALOG = DATA / "catalog.csv"         # the catalog produced in the cloud, for comparison

# ---- big generated data (6_Work) ----
WORK = PROJECT_ROOT / "6_Work"
RAW = WORK / "raw"                               # raw/<site>/<date>.tif   (written by collect.py --out WORK)
RAW_MANIFEST = WORK / "raw_manifest.csv"
CHIPS = WORK / "chips"                           # chips/<chip_id>.npz      (written by prepare.py --out WORK)
CATALOG = WORK / "catalog.csv"
NORM = WORK / "norm.json"
PREP_QA = WORK / "qa"
DATASETS = WORK / "datasets"
TRAIN_VAL = DATASETS / "Train_Val_Dataset"
TEST = DATASETS / "Test_Dataset"                 # LOCKED: never read by training or tuning code
LANDCOVER = WORK / "landcover"                   # landcover/raw/wc<year>_<site>.tif and landcover/<chip_id>.npz
TMP = WORK / "tmp"

# ---- results (4_Results) ----
RESULTS = PROJECT_ROOT / "4_Results"
BASELINES = RESULTS / "Baselines"


def unet_dir(preset: str = "cpu") -> Path:
    """Results folder of a U-Net run: 4_Results/Unet for the CPU preset, 4_Results/Unet_<preset> otherwise."""
    return RESULTS / ("Unet" if preset == "cpu" else f"Unet_{preset}")


def ensure_code_on_path() -> None:
    """Let scripts in subfolders (supporting_scripts/..., notebooks/) import chipio, evaluate, ..."""
    if str(CODE) not in sys.path:
        sys.path.insert(0, str(CODE))


def describe() -> str:
    return (f"project root : {PROJECT_ROOT}\n"
            f"inputs       : {DATA}\n"
            f"work (big)   : {WORK}\n"
            f"results      : {RESULTS}")


if __name__ == "__main__":
    print(describe())
