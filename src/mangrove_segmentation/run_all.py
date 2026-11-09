"""Run the whole pipeline, or any part of it, from the SDP folder.

Steps (in order)
  download    Sentinel-2 images listed in 3_Data/scenes_selected.csv  -> 6_Work/raw/       (10-20 min, ~2.4 GB)
  prepare     images + GMW labels -> 256 x 256 chips                   -> 6_Work/chips/     (~3 min)
  landcover   ESA WorldCover class per chip pixel                       -> 6_Work/landcover/ (~2 min)
  datasets    Train/Val dataset + LOCKED Test dataset                   -> 6_Work/datasets/
  baselines   B1 (NDVI/MNDWI threshold) and B2 (random forest), 5 folds -> 4_Results/Baselines (~25 min CPU)
  unet        U-Net, 5 folds (CUDA GPU used automatically if present)   -> 4_Results/Unet    (~45 min CPU)
  score_unet  U-Net vs baselines: tables, figures, report               -> 4_Results/Unet

Examples (from the SDP folder, in the Anaconda Prompt)
  python 2_Code\\run_all.py                              everything
  python 2_Code\\run_all.py --step download prepare      only these steps
  python 2_Code\\run_all.py --from datasets              this step and every later one
  python 2_Code\\run_all.py --skip landcover unet score_unet
  python 2_Code\\run_all.py --step unet score_unet --preset gpu
  python 2_Code\\run_all.py --sites eastern_mangroves ghantoot --dates 2020 --fast --epochs 2   quick trial
Every step skips work that is already done, so it is safe to run again after an interruption.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as P  # noqa: E402

STEPS = ["download", "prepare", "landcover", "datasets", "baselines", "unet", "score_unet"]
DATE = re.compile(r"_(\d{8})T")


def sh(script: Path, *args) -> None:
    cmd = [sys.executable, str(script), *map(str, args)]
    print("  $", " ".join(Path(c).name if i < 2 else c for i, c in enumerate(cmd)), flush=True)
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"}   # same text encoding on Windows; live output
    subprocess.run(cmd, check=True, cwd=P.PROJECT_ROOT, env=env)


def scenes_csv(sites, dates) -> Path:
    """3_Data/scenes_selected.csv, or a filtered copy in 6_Work/tmp when --sites / --dates are given."""
    if not sites and not dates:
        return P.SCENES
    import pandas as pd
    df = pd.read_csv(P.SCENES)
    if sites:
        missing = sorted(set(sites) - set(df.site))
        if missing:
            sys.exit(f"unknown site(s): {missing}. Sites: {sorted(df.site.unique())}")
        df = df[df.site.isin(sites)]
    if dates:
        d = df.item_id.str.extract(DATE)[0]
        pre = [x.replace("-", "") for x in dates]
        df = df[d.map(lambda v: any(str(v).startswith(p) for p in pre))]
    if df.empty:
        sys.exit("no scenes left after --sites/--dates")
    P.TMP.mkdir(parents=True, exist_ok=True)
    out = P.TMP / "scenes_subset.csv"
    df.to_csv(out, index=False)
    print(f"  subset: {len(df)} images, {df.site.nunique()} sites -> {out}")
    return out


def step_download(a):
    sh(P.CODE / "collect.py", "--sites", P.SITES, "--scenes", scenes_csv(a.sites, a.dates), "--out", P.WORK)
    print(f"  images -> {P.RAW}\n  list   -> {P.RAW_MANIFEST}")


def step_prepare(a):
    if not P.RAW_MANIFEST.exists():
        sys.exit(f"{P.RAW_MANIFEST} not found: run the download step first")
    sh(P.CODE / "prepare.py", "--manifest", P.RAW_MANIFEST, "--sites", P.SITES, "--labels", P.LABELS, "--out", P.WORK)
    print(f"  chips -> {P.CHIPS}\n  catalog -> {P.CATALOG}\n  QA figures -> {P.PREP_QA}")


def step_landcover(a):
    sh(P.CODE / "landcover.py", *(["--sites", *a.sites] if a.sites else []))


def step_datasets(a):
    sh(P.CODE / "make_datasets.py", *(["--rebuild"] if a.rebuild else []))


def step_baselines(a):
    b = P.CODE / "supporting_scripts" / "baselines"
    sh(b / "run_cv.py", *(["--fast"] if a.fast else []))
    sh(b / "score.py")
    print(f"  tables, figures, RESULTS -> {P.BASELINES}\n  predictions -> {P.WORK / 'runs' / 'Baselines'}")


def torch_ok() -> bool:
    try:
        import torch
    except ImportError:
        print("  PyTorch is not installed. In the Anaconda Prompt run:  pip install torch\n"
              "  (NVIDIA GPU: use the command from pytorch.org instead, see RUN_LOCALLY.md)")
        return False
    print(f"  PyTorch {torch.__version__}, CUDA GPU available: {torch.cuda.is_available()}")
    return True


def step_unet(a):
    if not torch_ok():
        sys.exit(1)
    args = ["--preset", a.preset]
    if a.epochs:
        args += ["--epochs", a.epochs]
    if a.device:
        args += ["--device", a.device]
    sh(P.CODE / "train_unet.py", *args)
    print(f"  per-fold scores, curves -> {P.unet_dir(a.preset)}\n  weights, predictions -> {P.WORK / 'runs' / P.unet_dir(a.preset).name}")


def step_score_unet(a):
    sh(P.CODE / "supporting_scripts" / "unet" / "score_unet.py", "--preset", a.preset)
    print(f"  report, figures -> {P.unet_dir(a.preset)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", nargs="+", choices=STEPS + ["all"], default=["all"], help="steps to run (default: all)")
    ap.add_argument("--from", dest="start", choices=STEPS, help="run this step and every later one")
    ap.add_argument("--skip", nargs="+", choices=STEPS, default=[], help="steps to leave out")
    ap.add_argument("--sites", nargs="+", help="only these sites (download, landcover); for quick trials")
    ap.add_argument("--dates", nargs="+", help="only image dates starting with these, e.g. 2020 or 202002 or 2020-11")
    ap.add_argument("--preset", choices=["cpu", "gpu"], default="cpu", help="U-Net size (default cpu)")
    ap.add_argument("--epochs", type=int, help="U-Net epochs (default: the preset's value)")
    ap.add_argument("--device", choices=["cpu", "cuda"], help="U-Net device (default: cuda if available)")
    ap.add_argument("--fast", action="store_true", help="baselines with small samples / few trees (trial only)")
    ap.add_argument("--rebuild", action="store_true", help="datasets: delete and rebuild 6_Work/datasets")
    a = ap.parse_args(argv)

    todo = STEPS if "all" in a.step else [s for s in STEPS if s in a.step]
    if a.start:
        todo = STEPS[STEPS.index(a.start):]
    todo = [s for s in todo if s not in a.skip]
    print(P.describe())
    print("steps:", ", ".join(todo), "\n")
    t_all = time.time()
    for i, s in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {s} ...", flush=True)
        t = time.time()
        globals()[f"step_{s}"](a)
        print(f"[{i}/{len(todo)}] {s} done in {(time.time() - t) / 60:.1f} min\n", flush=True)
    print(f"finished in {(time.time() - t_all) / 60:.1f} min. Results: {P.RESULTS}   Big files: {P.WORK}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        sys.exit(f"\nA step failed (exit code {e.returncode}). The error is printed above. "
                 "Fix it and run the same command again: finished work is kept.")
