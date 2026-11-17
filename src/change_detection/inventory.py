"""Which sites have 2018 and 2025 chips on the same chip grid?

Reads catalog.csv, hazy_scenes.csv and the chip files of the Train/Val dataset. Prints three tables and
saves them as CSV: one row per scene, the chip grid per site, and the labels stored in the 2018 and 2025 chips.

Usage: python inventory.py --train-val <Train_Val_Dataset> --out <output folder>
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import add_path_args, load_full_catalog


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_path_args(ap)
    a = ap.parse_args()
    tv, out = Path(a.train_val), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    cat = load_full_catalog(tv, a.qa)
    cat["cell"] = cat["row"].astype(str) + "_" + cat["col"].astype(str)
    print("catalog rows:", len(cat), "| roles:", sorted(cat["role"].unique()), "| sites:", sorted(cat["site"].unique()))

    # ---- one row per scene (site + date)
    scenes = cat.groupby(["site", "fold", "year", "date"]).agg(
        chips=("chip_id", "size"), water_b04_p5=("water_b04_p5", "first"), hazy=("hazy", "first"),
        chips_with_v4019_pct=("mangrove_pct_v4019", "count"), chips_with_v4112_pct=("mangrove_pct_v4112", "count"),
        max_cloud_pct=("cloud_pct", "max"), max_nodata_pct=("nodata_pct", "max")).reset_index()
    print("\nSCENES\n" + scenes.to_string(index=False))
    scenes.to_csv(out / "inventory_scenes.csv", index=False)

    # ---- chip grid per site: is a (row, col) cell always the same ground box, and which cells exist when?
    rows = []
    for site, g in cat.groupby("site"):
        same_box = bool(g.groupby("cell")["bounds_epsg32640"].nunique().max() == 1)
        by_date = {d: set(x["cell"]) for d, x in g.groupby("date")}

        def cells(year, how):
            sets = [by_date[d] for d in sorted(by_date) if d.startswith(str(year))]
            if not sets:
                return set()
            return set().union(*sets) if how == "any" else set.intersection(*sets)
        rows.append({"site": site, "fold": int(g["fold"].iloc[0]), "same_ground_box_per_cell": same_box,
                     "dates_2018": " ".join(d for d in sorted(by_date) if d.startswith("2018")),
                     "dates_2025": " ".join(d for d in sorted(by_date) if d.startswith("2025")),
                     "cells_any_date": len(set(g["cell"])),
                     "cells_2018_any": len(cells(2018, "any")), "cells_2025_any": len(cells(2025, "any")),
                     "cells_in_2018_and_2025": len(cells(2018, "any") & cells(2025, "any")),
                     "cells_in_all_2018_and_2025_dates": len(cells(2018, "all") & cells(2025, "all")),
                     "cells_2020_any": len(cells(2020, "any")),
                     "cells_18_25_also_in_2020": len(cells(2018, "any") & cells(2025, "any") & cells(2020, "any"))})
    grid = pd.DataFrame(rows).sort_values("fold")
    print("\nGRID\n" + grid.to_string(index=False))
    grid.to_csv(out / "inventory_grid.csv", index=False)

    # ---- labels actually stored in the 2018 / 2025 chips
    lab = []
    for _, r in cat[cat["year"].isin([2018, 2025])].iterrows():
        with np.load(tv / r["path"], allow_pickle=False) as z:
            v19, v12 = z["label_v4019"], z["label_v4112"]
            lab.append({"site": r["site"], "date": r["date"], "year": r["year"],
                        "v4019_labelled_px": int((v19 != 255).sum()), "v4112_labelled_px": int((v12 != 255).sum()),
                        "v4112_mangrove_px": int((v12 == 1).sum())})
    lab = pd.DataFrame(lab).groupby(["site", "year", "date"]).sum().reset_index()
    print("\nLABELS IN 2018 / 2025 CHIPS (pixel counts)\n" + lab.to_string(index=False))
    lab.to_csv(out / "inventory_labels.csv", index=False)


if __name__ == "__main__":
    main()
