"""Same-year noise floor for every Train/Val site, each with the random forest that never saw it.

For each of the 5 folds the forest is trained with that fold held out (common.train_fold, variant b by default).
The held-out sites' non-hazy scenes are predicted. Then, per site:
  * noise floor: every pair of dates of the same year is compared as if it were a change map
    (single-date masks, pixels usable on both dates). The net "change" should be about zero.
  * the 2018 to 2025 change, made four ways, to see whether it is larger than the site's own noise.
  * the site's held-out IoU on its 2020 chips against GMW v4.0.19.

Writes to <out>/all_sites/: noise_floor_all_pairs.csv, change_2018_2025_all_sites.csv, site_summary.csv, fold_fits.csv.

Usage: python noise_floor_all_sites.py --train-val <Train_Val_Dataset> --out <output folder>
"""
import argparse
import itertools
import json
from pathlib import Path

import pandas as pd

from change_map import composite, change_stats, pair_stats, site_scenes
from common import CHIP, add_path_args, heldout_scores, load_full_catalog, published_fold_iou, train_fold


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_path_args(ap)
    ap.add_argument("--variant", default="b", choices=["a", "b"])
    a = ap.parse_args()
    tv = Path(a.train_val)
    out = Path(a.out) / "all_sites"
    out.mkdir(parents=True, exist_ok=True)

    cat = load_full_catalog(tv, a.qa)
    cat20 = cat[cat["year"] == 2020].reset_index(drop=True)
    noise, cross, fits = [], [], []
    for k in sorted(int(f) for f in cat20.fold.unique()):
        fit = train_fold(cat20, k, tv, out / "work", a.variant)
        rf, th = fit["rf"], fit["threshold"]
        scores = heldout_scores(rf, th, fit["held"], tv)
        fold_iou = round(scores["__fold__"]["iou"], 4)
        print(f"fold {k}: threshold {th}, settings {fit['settings']}, held-out IoU {fold_iou} "
              f"(committed B2 result {published_fold_iou(k, a.variant)})", flush=True)
        for site in fit["info"]["held_sites"]:
            site_iou = round(scores[site]["iou"], 4)
            fits.append({"fold": k, "site": site, "threshold": th, "settings": json.dumps(fit["settings"]),
                         "train_chips": fit["info"]["train_chips"], "heldout_iou_site": site_iou,
                         "heldout_iou_fold": fold_iou, "committed_b2_fold_iou": published_fold_iou(k, a.variant)})
            scenes, hazy, _, _, _ = site_scenes(cat, site, rf, th, tv)
            by_year = {y: sorted(d for d in scenes if d.startswith(str(y))) for y in (2018, 2020, 2025)}
            base = {"site": site, "fold": k, "heldout_iou": site_iou}

            # ---- same-year pairs (every pair of non-hazy dates)
            for y, ds in by_year.items():
                for d0, d1 in itertools.combinations(ds, 2):
                    row = {**base, "year": y, **pair_stats(scenes, d0, d1)}
                    if row["chips_compared"] > 0:
                        noise.append(row)

            # ---- 2018 -> 2025, four ways
            if by_year[2018] and by_year[2025]:
                m18, u18, _ = composite([scenes[d] for d in by_year[2018]], th)
                m25, u25, _ = composite([scenes[d] for d in by_year[2025]], th)
                use = u18 & u25
                n18 = sum(scenes[d]["valid"].astype(int) for d in by_year[2018])
                n25 = sum(scenes[d]["valid"].astype(int) for d in by_year[2025])
                cross.append({**base, "version": "vote over all dates", "dates": " ".join(by_year[2018] + by_year[2025]),
                              "chips_compared": int(round(use.sum() / CHIP ** 2)), **change_stats(m18, m25, use)})
                both = (n18 >= 2) & (n25 >= 2)
                if both.any():
                    cross.append({**base, "version": "vote, only pixels with two dates in both years", "dates": "",
                                  "chips_compared": int(round(both.sum() / CHIP ** 2)), **change_stats(m18, m25, both)})
                for d0, d1 in itertools.product(by_year[2018], by_year[2025]):
                    if d0[5:7] == d1[5:7]:          # same month in both years
                        row = pair_stats(scenes, d0, d1)
                        if row["chips_compared"] > 0:
                            row = {c: v for c, v in row.items() if c not in ("first_date", "second_date")}
                            cross.append({**base, "version": "single dates, same month", "dates": f"{d0} {d1}", **row})
            print(f"  {site}: {len(scenes)} scenes predicted, hazy skipped: {hazy}", flush=True)

    noise = pd.DataFrame(noise); cross = pd.DataFrame(cross); fits = pd.DataFrame(fits)
    noise.to_csv(out / "noise_floor_all_pairs.csv", index=False)
    cross.to_csv(out / "change_2018_2025_all_sites.csv", index=False)
    fits.to_csv(out / "fold_fits.csv", index=False)

    # ---- per site: how big is the noise, how big is the 2018 -> 2025 change
    summ = []
    for site, g in noise.groupby("site"):
        c = cross[cross["site"] == site]
        vote = c[c["version"] == "vote over all dates"]
        sym = c[c["version"] == "vote, only pixels with two dates in both years"]
        single = c[c["version"] == "single dates, same month"]
        summ.append({"site": site, "heldout_iou": g["heldout_iou"].iloc[0], "same_year_pairs": len(g),
                     "noise_max_abs_net_pct": round(g["net_pct_of_first"].abs().max(), 1),
                     "noise_median_abs_net_pct": round(g["net_pct_of_first"].abs().median(), 1),
                     "noise_net_pct_list": " ".join(f"{v:+.1f}" for v in g["net_pct_of_first"]),
                     "min_chips_in_a_pair": int(g["chips_compared"].min()),
                     "change_vote_net_pct": float(vote["net_pct_of_first"].iloc[0]) if len(vote) else None,
                     "change_vote_chips": int(vote["chips_compared"].iloc[0]) if len(vote) else None,
                     "change_two_date_pixels_net_pct": float(sym["net_pct_of_first"].iloc[0]) if len(sym) else None,
                     "change_single_date_net_pct_list": " ".join(f"{v:+.1f}" for v in single["net_pct_of_first"])})
    summ = pd.DataFrame(summ).sort_values("noise_max_abs_net_pct")
    summ.to_csv(out / "site_summary.csv", index=False)

    with pd.option_context("display.width", 250):
        print("\nSAME-YEAR PAIRS\n" + noise[["site", "year", "first_date", "second_date", "chips_compared", "area_first_ha", "gain_ha",
                                             "loss_ha", "net_pct_of_first", "changed_pct_of_first", "heldout_iou"]].to_string(index=False))
        print("\n2018 -> 2025\n" + cross[["site", "version", "dates", "chips_compared", "area_first_ha", "area_second_ha", "gain_ha",
                                          "loss_ha", "net_ha", "net_pct_of_first"]].to_string(index=False))
        print("\nSITE SUMMARY\n" + summ.to_string(index=False))
        print("\nFITS\n" + fits.to_string(index=False))


if __name__ == "__main__":
    main()
