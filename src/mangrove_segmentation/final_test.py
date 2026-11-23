"""The single final test of the mangrove model. Read results/mangrove_segmentation/final_test/PLAN.md first.

The model is frozen: random forest, variant (b), max_depth 12, min_samples_leaf 10, 200 trees,
200,000 class-balanced pixels, seed 0, threshold 0.85. Nothing is tuned in this script.
The forest code, the chip reading and the scoring are imported unchanged from baseline_rf.py,
chipio.py and evaluate.py. No chip is plotted anywhere.

Three modes:
    python final_test.py fit
        Trains the frozen forest on the 2020 Train/Val chips (hazy scenes dropped) and saves it
        outside the repository with its sha256.
    python final_test.py --dry-run
        Runs the whole predict and score path on Train/Val chips only, to prove the code works.
        The scores it prints are training-set scores. They are not reported anywhere.
    python final_test.py --run-test --test-root <Test_Dataset>
        The real test. Expects <Test_Dataset>/fixed_test and <Test_Dataset>/external, each with
        catalog.csv and a chips folder. It runs once. It refuses to run if TEST_RUN_DONE.json is
        already in the output folder, if the git working tree is dirty, if PLAN.md is not
        committed, or if the model file is not the one whose sha256 is written in PLAN.md.

Everything is written under --out (default C:/Users/Zain/Documents/sdp-results/final_test), outside the repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier

import baseline_rf as B2          # the team's forest code, unchanged
import evaluate as EV             # the team's scoring code, unchanged

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PLAN_REL = "results/mangrove_segmentation/final_test/PLAN.md"

DEFAULT_TRAIN_VAL = Path(r"C:\Users\Zain\Documents\sdp-data\Train_Val_Dataset")
DEFAULT_OUT = Path(r"C:\Users\Zain\Documents\sdp-results\final_test")

# ---- the frozen spec (PLAN.md). Do not change after the plan is committed. ----
SETTINGS = {"max_depth": 12, "min_samples_leaf": 10}   # picked by all 5 folds
TREES = 200
CAP = 200_000            # class-balanced training pixels, up to half per class
SEED = 0
THRESHOLD = 0.85         # median of the five fold thresholds 0.85, 0.90, 0.85, 0.85, 0.80
LABEL = "v4019"          # GMW v4.0.19
HAZE_P5 = 0.05           # dark-water red above this = hazy scene (same rule as run_cv.py and make_datasets.py)
TRAIN_SITES = ["al_aryam_islands", "al_nouf", "eastern_mangroves", "ghantoot", "jubail_island", "ras_ghurab"]
TEST_GROUPS = {"fixed_test": 2, "external": 5}   # folder -> number of locked sites (make_datasets.py)

# cross-validation numbers the test is reported next to (results/mangrove_segmentation/baselines/RESULTS.md)
CV_REFERENCE = {"mean_fold_iou": 0.704, "sd_fold_iou": 0.101, "pooled_oof_iou": 0.686, "pooled_oof_iou_ci95": [0.653, 0.713]}

DRY_BANNER = "DRY RUN. TRAINING-SET SCORES. NOT A TEST RESULT. NOT REPORTED ANYWHERE."


# ------------------------------------------------------------------ small helpers
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git(*args: str) -> tuple[int, str]:
    r = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()


def plan_model_sha() -> str | None:
    """The model sha256 written in PLAN.md, or None if PLAN.md has none."""
    p = REPO / PLAN_REL
    if not p.exists():
        return None
    m = re.search(r"sha256[^0-9a-f\n]*([0-9a-f]{64})", p.read_text(encoding="utf-8"))
    return m.group(1) if m else None


def model_paths(out: Path) -> tuple[Path, Path]:
    return out / "model" / "rf_final.joblib", out / "model" / "rf_final.json"


def frozen_meta() -> dict:
    return {"model": "random_forest", "variant": "b_no_haze", "label": LABEL, "seed": SEED, "n_estimators": TREES,
            "settings": SETTINGS, "cap": CAP, "threshold": THRESHOLD}


def load_model(out: Path) -> tuple[dict, str]:
    """Load the saved forest and check it is the frozen one. Returns (bundle, sha256 of the file)."""
    mp, jp = model_paths(out)
    if not mp.exists():
        raise SystemExit(f"no model at {mp}. Run: python final_test.py fit")
    sha = sha256_of(mp)
    recorded = json.loads(jp.read_text())["sha256"]
    if sha != recorded:
        raise SystemExit(f"REFUSED: {mp.name} has sha256 {sha} but {jp.name} says {recorded}")
    bundle = joblib.load(mp)
    for k, v in frozen_meta().items():
        if bundle["meta"].get(k) != v:
            raise SystemExit(f"REFUSED: saved model has {k}={bundle['meta'].get(k)!r}, the frozen spec says {v!r}")
    return bundle, sha


# ------------------------------------------------------------------ fit
def training_catalog(tv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(2020 chips used for training, 2020 chips dropped as hazy), in catalog order."""
    cat = pd.read_csv(tv / "catalog.csv")
    if sorted(cat["site"].unique()) != TRAIN_SITES:
        raise SystemExit(f"REFUSED: {tv} does not hold exactly the six Train/Val sites: {sorted(cat['site'].unique())}")
    if "role" in cat.columns and set(cat["role"]) != {"cv"}:
        raise SystemExit(f"REFUSED: {tv}/catalog.csv has chips whose role is not 'cv'")
    cat20 = cat[cat["year"] == 2020].copy()
    qa = pd.read_csv(tv / "hazy_scenes.csv")[["site", "date", "water_b04_p5"]].drop_duplicates(["site", "date"])
    cat20 = cat20.merge(qa, on=["site", "date"], how="left")
    if cat20["water_b04_p5"].isna().any():
        raise SystemExit("some 2020 scenes have no haze value in hazy_scenes.csv")
    cat20["hazy"] = cat20["water_b04_p5"] > HAZE_P5
    cat20 = cat20.reset_index(drop=True)
    return cat20[~cat20["hazy"]], cat20[cat20["hazy"]]


def fit(tv: Path, out: Path) -> None:
    mp, jp = model_paths(out)
    if (out / "test_run" / "TEST_RUN_DONE.json").exists():
        raise SystemExit("REFUSED: the test has already run. The model stays as it is.")
    if mp.exists():
        raise SystemExit(f"REFUSED: {mp} already exists. The model is frozen. Its sha256 is in {jp.name} and in PLAN.md.")
    t0 = time.time()
    train, hazy = training_catalog(tv)
    print(f"2020 Train/Val chips: {len(train) + len(hazy)}. Hazy and dropped: {len(hazy)}. Used for training: {len(train)}.", flush=True)
    # the same three lines as the final refit in run_cv.py, on all six sites
    rng = np.random.default_rng(SEED)
    X, y = B2.collect(train, tv, LABEL, CAP, rng, balanced=True)
    rf = RandomForestClassifier(n_estimators=TREES, n_jobs=-1, random_state=SEED, **SETTINGS).fit(X, y)

    meta = {**frozen_meta(),
            "train_chips": int(len(train)), "hazy_chips_dropped": int(len(hazy)),
            "hazy_scenes_dropped": sorted(f"{s} {d}" for s, d in hazy[["site", "date"]].drop_duplicates().itertuples(index=False)),
            "train_chips_per_site": {s: int(n) for s, n in train["site"].value_counts().sort_index().items()},
            "train_pixels": int(len(y)), "train_pixels_mangrove": int(y.sum()),
            "features": ["B02", "B03", "B04", "B08", "B11", "B12", "NDVI", "MNDWI"],
            "feature_importances": [round(float(v), 4) for v in rf.feature_importances_],
            "train_val_folder": str(tv), "fitted_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sklearn": sklearn.__version__, "numpy": np.__version__, "python": sys.version.split()[0]}
    mp.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"rf": rf, "meta": meta}, mp, compress=3)   # same bundle shape as baseline_rf.py, so B2.predict reads it
    sha = sha256_of(mp)
    jp.write_text(json.dumps({**meta, "sha256": sha, "file": mp.name}, indent=1))
    (mp.parent / "rf_final.sha256").write_text(f"{sha}  {mp.name}\n")
    print(f"training pixels: {len(y)} ({int(y.sum())} mangrove). Fit took {time.time() - t0:.0f} s.")
    print(f"saved  {mp}")
    print(f"sha256 {sha}")
    print(f"Write this sha256 into {PLAN_REL} before committing the plan.")


# ------------------------------------------------------------------ predict, score, tables
def check_catalogs(groups: dict[str, tuple[Path, Path]], expected_sites: dict[str, int], forbidden: set[str]) -> dict:
    """Checks on the catalogs before any chip is predicted. Raises SystemExit, so nothing is scored on a failure."""
    info, seen = {}, set()
    for name, (catalog, _) in groups.items():
        cat = EV.load_catalog(catalog)          # needs chip_id, site, split
        sites = sorted(cat["site"].unique())
        bad = sorted(set(sites) & forbidden)
        if bad:
            raise SystemExit(f"REFUSED: {name} holds Train/Val sites {bad}. No chip was predicted.")
        if len(sites) != expected_sites[name]:
            raise SystemExit(f"REFUSED: {name} should hold {expected_sites[name]} sites but its catalog has "
                             f"{len(sites)}: {sites}. No chip was predicted.")
        dup = seen & set(cat["chip_id"])
        if dup or cat["chip_id"].duplicated().any():
            raise SystemExit(f"REFUSED: {name} has chip ids that appear twice. No chip was predicted.")
        seen |= set(cat["chip_id"])
        info[name] = {"sites": sites, "chips_in_catalog": int(len(cat))}
    return info


def predict_group(bundle: dict, catalog: Path, chips_dir: Path, pred_dir: Path) -> int:
    """Masks for every chip in the catalog, with baseline_rf.predict as it is. No score is computed here."""
    return B2.predict(catalog, chips_dir, bundle, "all", pred_dir)


def score_group(catalog: Path, chips_dir: Path, pred_dir: Path) -> tuple[dict, pd.DataFrame]:
    """evaluate.py as it is: label v4019, mangrove class, per site and pooled, 1000 bootstrap reps over chips, seed 0."""
    return EV.evaluate(pred_dir, catalog, chips_dir, split="all", label=LABEL)


def table_rows(scores: dict) -> pd.DataFrame:
    rows = []
    for name, (res, _) in scores.items():
        for site, r in list(res["per_site"].items()) + [("pooled", res["pooled"])]:
            row = {"group": name, "site": site, "chips": r["n_chips"], "mangrove_px": r["mangrove_px"]}
            for m in EV.METRICS:
                row[m] = r[m]; row[f"{m}_lo"], row[f"{m}_hi"] = r[f"{m}_ci95"]
            rows.append(row)
    return pd.DataFrame(rows)


def tables_md(scores: dict, title: str, banner: str | None) -> str:
    lines = [f"# {title}", ""]
    if banner:
        lines += [f"**{banner}**", ""]
    lines += ["Random forest, variant (b), threshold 0.85. Label GMW v4.0.19, mangrove class. Scored with `evaluate.py`: "
              "pixels pooled per site and over all chips, 95% interval from 1000 bootstrap draws of chips (seed 0).", ""]
    for name, (res, _) in scores.items():
        lines += [f"## {name}", "",
                  "| Site | Chips | Mangrove px | IoU [95% CI] | F1 [95% CI] | Precision [95% CI] | Recall [95% CI] |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for site, r in list(res["per_site"].items()) + [("**pooled**", res["pooled"])]:
            lines.append(f"| {site} | {r['n_chips']} | {r['mangrove_px']} | " + " | ".join(EV.fmt(r, m) for m in EV.METRICS) + " |")
        lines += ["", f"Chips with no v4019 label, not scored: {len(res['chips_skipped_no_label'])}.", ""]
    c = CV_REFERENCE
    lines += ["## Cross-validation, for comparison", "",
              f"Random forest (b) on Train/Val, leave one site out: mean IoU over the 5 folds {c['mean_fold_iou']:.3f} "
              f"(sd {c['sd_fold_iou']:.3f}). Pixels pooled over all held-out chips: {c['pooled_oof_iou']:.3f} "
              f"[{c['pooled_oof_iou_ci95'][0]:.3f}, {c['pooled_oof_iou_ci95'][1]:.3f}]. "
              "The pooled rows above are pixel-pooled, so the second number is the like-for-like one.", ""]
    return "\n".join(lines)


def write_outputs(run_dir: Path, scores: dict, header: dict, title: str, banner: str | None) -> None:
    for name, (res, per_chip) in scores.items():
        d = run_dir / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "results.json").write_text(json.dumps({**res, "name": f"rf_final_{name}"}, indent=1))
        per_chip.to_csv(d / "per_chip.csv", index=False)
        (d / "results.md").write_text(EV.markdown_table(res, f"rf_final {name}"), encoding="utf-8")
    combined = {**header, "frozen_spec": frozen_meta(), "cv_reference": CV_REFERENCE,
                "groups": {name: res for name, (res, _) in scores.items()}}
    (run_dir / "results.json").write_text(json.dumps(combined, indent=1))
    table_rows(scores).round(4).to_csv(run_dir / "tables.csv", index=False)
    (run_dir / "tables.md").write_text(tables_md(scores, title, banner), encoding="utf-8")


def preflight(run_dir: Path, model_sha: str) -> list[str]:
    """Reasons to refuse the real test. An empty list means it may run."""
    why = []
    if (run_dir / "TEST_RUN_DONE.json").exists():
        why.append(f"TEST_RUN_DONE.json already exists in {run_dir}. The test runs once.")
    code, status = git("status", "--porcelain")
    if code != 0:
        why.append(f"git status failed: {status}")
    elif status:
        why.append("the git working tree is dirty:\n" + status)
    if git("cat-file", "-e", f"HEAD:{PLAN_REL}")[0] != 0:
        why.append(f"{PLAN_REL} is not committed.")
    pinned = plan_model_sha()
    if pinned is None:
        why.append("PLAN.md does not give the model sha256.")
    elif pinned != model_sha:
        why.append(f"the model file has sha256 {model_sha} but PLAN.md says {pinned}.")
    return why


# ------------------------------------------------------------------ the two run modes
def dry_run(tv: Path, out: Path) -> None:
    print("=" * 78); print(DRY_BANNER); print("=" * 78, flush=True)
    bundle, sha = load_model(out)
    run_dir = out / "dry_run"
    why = preflight(out / "test_run", sha)
    print("Checks the real test will make (only shown here, a dry run never refuses):")
    print("  would run" if not why else "  would REFUSE:\n    " + "\n    ".join(w.replace("\n", "\n      ") for w in why), flush=True)

    # stand-ins for the two test folders, made from Train/Val chips of all years (2018 and 2025 chips have no v4019 label)
    cat = pd.read_csv(tv / "catalog.csv")
    stand_in = {"stand_in_fixed_test": ["eastern_mangroves", "ghantoot"],
                "stand_in_external": ["al_aryam_islands", "al_nouf", "jubail_island", "ras_ghurab"]}
    groups = {}
    for name, sites in stand_in.items():
        d = run_dir / name
        d.mkdir(parents=True, exist_ok=True)
        cat[cat["site"].isin(sites)].to_csv(d / "catalog.csv", index=False)
        groups[name] = (d / "catalog.csv", tv)
    info = check_catalogs(groups, {k: len(v) for k, v in stand_in.items()}, forbidden=set())
    t0 = time.time()
    for name, (catalog, chips) in groups.items():
        n = predict_group(bundle, catalog, chips, run_dir / "preds" / name)
        print(f"{name}: predicted {n} Train/Val chips ({time.time() - t0:.0f} s)", flush=True)
    scores = {name: score_group(catalog, chips, run_dir / "preds" / name) for name, (catalog, chips) in groups.items()}
    header = {"dry_run": True, "note": DRY_BANNER, "model_sha256": sha, "catalogs": info,
              "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    write_outputs(run_dir, scores, header, "Dry run on Train/Val chips", DRY_BANNER)
    (run_dir / "DRY_RUN_DONE.json").write_text(json.dumps({**header, "git_commit": git("rev-parse", "HEAD")[1]}, indent=1))
    print()
    print((run_dir / "tables.md").read_text(encoding="utf-8"))
    print("=" * 78); print(DRY_BANNER)
    print("The forest was trained on these chips, so these numbers say only that the code runs.")
    print(f"written to {run_dir}"); print("=" * 78)


def run_test(test_root: Path, tv: Path, out: Path) -> None:
    run_dir = out / "test_run"
    bundle, sha = load_model(out)
    why = preflight(run_dir, sha)
    if why:
        raise SystemExit("REFUSED to run the test:\n- " + "\n- ".join(why))
    if test_root.resolve() == tv.resolve():
        raise SystemExit("REFUSED: --test-root is the Train/Val folder.")
    commit = git("rev-parse", "HEAD")[1]
    groups = {name: (test_root / name / "catalog.csv", test_root / name) for name in TEST_GROUPS}
    for name, (catalog, chips) in groups.items():
        if not catalog.exists() or not (chips / "chips").is_dir():
            raise SystemExit(f"REFUSED: expected {catalog} and {chips / 'chips'}. No chip was predicted.")
    info = check_catalogs(groups, TEST_GROUPS, forbidden=set(TRAIN_SITES))
    run_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for name, (catalog, chips) in groups.items():      # masks only, no score yet
        n = predict_group(bundle, catalog, chips, run_dir / "preds" / name)
        print(f"{name}: predicted {n} chips", flush=True)

    scores, error = {}, None
    try:
        for name, (catalog, chips) in groups.items():
            scores[name] = score_group(catalog, chips, run_dir / "preds" / name)
    except BaseException as e:          # also SystemExit from evaluate.py
        error = f"{type(e).__name__}: {e}"
    if scores:                          # a score exists: the one run is used up, whatever happens next
        done = {"time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "started_utc": started,
                "git_commit": commit, "model_sha256": sha, "test_root": str(test_root),
                "groups_scored": list(scores), "catalogs": info, "error": error}
        (run_dir / "TEST_RUN_DONE.json").write_text(json.dumps(done, indent=1))
        header = {"dry_run": False, "model_sha256": sha, "git_commit": commit, "catalogs": info, "time_utc": done["time_utc"]}
        write_outputs(run_dir, scores, header, "Final test of the random forest", None)
        print()
        print((run_dir / "tables.md").read_text(encoding="utf-8"))
        print(f"written to {run_dir}. TEST_RUN_DONE.json is in place. The test does not run again.")
    if error:
        raise SystemExit(f"scoring stopped: {error}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", nargs="?", choices=["fit"], help="fit = train and save the frozen forest")
    ap.add_argument("--dry-run", action="store_true", help="predict and score Train/Val chips only")
    ap.add_argument("--run-test", action="store_true", help="the real test. Runs once.")
    ap.add_argument("--test-root", help="folder holding fixed_test and external (only with --run-test)")
    ap.add_argument("--train-val", default=str(DEFAULT_TRAIN_VAL), help="the Train_Val_Dataset folder")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output folder, outside the repository")
    a = ap.parse_args()
    if (a.mode == "fit") + a.dry_run + a.run_test != 1:
        ap.error("choose exactly one of: fit, --dry-run, --run-test")
    if a.test_root and not a.run_test:
        ap.error("--test-root is only used with --run-test")
    tv, out = Path(a.train_val), Path(a.out)
    if a.mode == "fit":
        fit(tv, out)
    elif a.dry_run:
        dry_run(tv, out)
    else:
        if not a.test_root:
            ap.error("--run-test needs --test-root")
        run_test(Path(a.test_root), tv, out)


if __name__ == "__main__":
    main()
