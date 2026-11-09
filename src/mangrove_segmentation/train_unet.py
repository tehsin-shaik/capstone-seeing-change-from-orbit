"""Stage 5 (M1): small U-Net with 5-fold leave-one-site-out cross-validation.

Same protocol as the baselines (results/baselines/scripts/run_cv.py):
  * Train_Val_Dataset only, 2020 chips, label GMW v4.0.19 (10 m). Test_Dataset is never read.
  * Outer fold k = one Abu Dhabi site (fold 5 = Al Aryam + Al Nouf) held out.
  * Training pool = the other 4 folds, hazy scenes dropped (variant b: dark-water red p5 > 0.05),
    minus any chip within one chip (2,560 m) of a held-out chip (Eastern Mangroves / Jubail buffer).
  * Inner validation fold = the same one baseline B2 used (first of 3,4,5,2,1 that is not k), also
    buffered. It drives early stopping and the decision threshold. The U-Net is NOT refitted on all 4
    folds afterwards (B2 was): it trains on 3 folds. This is a documented handicap of the CPU version.
  * Held-out fold predicted on full 256 x 256 chips, scored with code/evaluate.py exactly like B1/B2.

Usage (from the SDP folder, or via run_all.py --step unet):
  python 2_Code/train_unet.py --preset cpu         # what was run on 2 CPU cores -> 4_Results/Unet
  python 2_Code/train_unet.py --preset gpu         # NVIDIA GPU / Colab          -> 4_Results/Unet_gpu
  python 2_Code/train_unet.py --folds 2 3          # only some folds; earlier folds' rows are kept
  python 2_Code/train_unet.py --bench              # time a few training steps and exit
Paths come from paths.py. Uses the GPU automatically when PyTorch can see one (--device cpu to force CPU).
Model weights and predictions go to 6_Work/runs/<results folder name>; tables and figures to 4_Results.
"""
from __future__ import annotations

import argparse, json, math, random, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths as P  # noqa: E402
from chipio import chip_path, get_label, load_chip, save_mask  # noqa: E402
import evaluate as EV  # noqa: E402

TV = P.TRAIN_VAL                                      # notebooks may override TV and QA_CSV
QA_CSV = P.QA_CSV
DEVICE = torch.device("cpu")                          # set in run() / main
HAZE_P5 = 0.05
BUFFER_M = 2560
INNER_PREF = [3, 4, 5, 2, 1]
REFL = slice(0, 6)          # B02 B03 B04 B08 B11 B12
INDEX_EPS = 0.01

PRESETS = {
    "cpu": dict(base=16, crop=128, crops_per_chip=3, batch=16, epochs=40, patience=8, lr=1e-3),
    "gpu": dict(base=32, crop=256, crops_per_chip=1, batch=16, epochs=100, patience=15, lr=1e-3),
}


# ----------------------------------------------------------------------------- data
def load_cat() -> pd.DataFrame:
    cat = pd.read_csv(TV / "catalog.csv")
    cat = cat[cat["year"] == 2020].copy()
    qa = pd.read_csv(QA_CSV)[["site", "date", "water_b04_p5"]]
    cat = cat.merge(qa, on=["site", "date"], how="left")
    cat["hazy"] = cat["water_b04_p5"] > HAZE_P5
    b = cat["bounds_epsg32640"].str.split(",", expand=True).astype(float)
    cat[["x0", "y0", "x1", "y1"]] = b.values
    cat["fold"] = cat["fold"].astype(int)
    return cat.reset_index(drop=True)


def drop_near(train: pd.DataFrame, other: pd.DataFrame, buf=BUFFER_M) -> pd.DataFrame:
    if train.empty or other.empty:
        return train
    keep = []
    for _, r in train.iterrows():
        dx = np.maximum(0, np.maximum(other.x0.values - r.x1, r.x0 - other.x1.values))
        dy = np.maximum(0, np.maximum(other.y0.values - r.y1, r.y0 - other.y1.values))
        keep.append(np.maximum(dx, dy).min() >= buf)
    return train[np.array(keep)]


def load_all(cat: pd.DataFrame) -> dict:
    """chip_id -> (image float32 [8,H,W] with NaN, target uint8 [H,W], usable bool [H,W])."""
    out = {}
    for _, r in cat.iterrows():
        c = load_chip(chip_path(TV, r))
        truth, labelled = get_label(c, "v4019")
        out[r.chip_id] = (c["image"].astype(np.float32), truth.astype(np.uint8), (c["valid"] & labelled))
    return out


def fold_stats(ids, data):
    s = np.zeros(8); ss = np.zeros(8); n = 0
    for i in ids:
        img, _, use = data[i]
        x = img[:, use].astype(np.float64)
        s += x.sum(1); ss += (x ** 2).sum(1); n += x.shape[1]
    mean = s / n
    return mean.astype(np.float32), np.sqrt(np.maximum(ss / n - mean ** 2, 1e-12)).astype(np.float32)


def recompute_indices(img):
    b03, b04, b08, b11 = img[1], img[2], img[3], img[4]
    img[6] = np.clip((b08 - b04) / np.maximum(b08 + b04, INDEX_EPS), -1, 1)
    img[7] = np.clip((b03 - b11) / np.maximum(b03 + b11, INDEX_EPS), -1, 1)
    return img


def normalise(img, mean, std):
    x = (img - mean[:, None, None]) / std[:, None, None]
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def make_batch(ids, data, mean, std, crop, rng, augment=True):
    xs, ys, ms = [], [], []
    for i in ids:
        img, y, use = data[i]
        H, W = y.shape
        r0 = rng.integers(0, H - crop + 1); c0 = rng.integers(0, W - crop + 1)
        im = img[:, r0:r0 + crop, c0:c0 + crop].copy(); yy = y[r0:r0 + crop, c0:c0 + crop]; mm = use[r0:r0 + crop, c0:c0 + crop]
        if augment:
            im[REFL] += rng.uniform(-0.02, 0.02)
            im = recompute_indices(im)
            k = rng.integers(0, 4)
            im = np.rot90(im, k, axes=(1, 2)); yy = np.rot90(yy, k); mm = np.rot90(mm, k)
            if rng.random() < 0.5:
                im = im[:, :, ::-1]; yy = yy[:, ::-1]; mm = mm[:, ::-1]
        xs.append(normalise(im, mean, std)); ys.append(yy.copy()); ms.append(mm.copy())
    return (torch.from_numpy(np.stack(xs)).float(), torch.from_numpy(np.stack(ys)).float(),
            torch.from_numpy(np.stack(ms)).float())


# ----------------------------------------------------------------------------- model
def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNet(nn.Module):
    """4 levels: base, 2b, 4b, 8b, bottleneck 16b; outputs one logit per pixel."""
    def __init__(self, cin=8, base=16):
        super().__init__()
        ch = [base, base * 2, base * 4, base * 8]
        self.enc = nn.ModuleList()
        prev = cin
        for c in ch:
            self.enc.append(block(prev, c)); prev = c
        self.mid = block(ch[-1], ch[-1] * 2)
        self.up = nn.ModuleList(); self.dec = nn.ModuleList()
        prev = ch[-1] * 2
        for c in reversed(ch):
            self.up.append(nn.ConvTranspose2d(prev, c, 2, stride=2))
            self.dec.append(block(c * 2, c)); prev = c
        self.head = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x):
        skips = []
        for e in self.enc:
            x = e(x); skips.append(x); x = F.max_pool2d(x, 2)
        x = self.mid(x)
        for u, d, s in zip(self.up, self.dec, reversed(skips)):
            x = d(torch.cat([u(x), s], 1))
        return self.head(x)[:, 0]


def loss_fn(logits, y, m, pos_weight=4.0):
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none",
                                             pos_weight=torch.tensor(pos_weight, device=logits.device))
    bce = (bce * m).sum() / m.sum().clamp(min=1)
    p = torch.sigmoid(logits) * m
    inter = (p * y * m).sum(); dice = 1 - (2 * inter + 1) / (p.sum() + (y * m).sum() + 1)
    return bce + dice


@torch.no_grad()
def predict_prob(model, ids, data, mean, std, batch=8):
    model.eval(); out = {}
    for j in range(0, len(ids), batch):
        b = ids[j:j + batch]
        x = torch.from_numpy(np.stack([normalise(data[i][0], mean, std) for i in b])).float().to(DEVICE)
        p = torch.sigmoid(model(x)).cpu().numpy()
        for i, pp in zip(b, p):
            out[i] = pp
    return out


def iou_at(probs, ids, data, t):
    tp = fp = fn = 0
    for i in ids:
        _, y, use = data[i]; pr = probs[i] > t; yy = y.astype(bool)
        tp += np.sum(pr & yy & use); fp += np.sum(pr & ~yy & use); fn += np.sum(~pr & yy & use)
    return tp / max(tp + fp + fn, 1)


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def pick_inner(k, present):
    """Inner-validation fold: the first of INNER_PREF that is not the held-out fold (as B2 used).
    With every fold present this is exactly the original rule; on a subset it skips missing folds."""
    return next(f for f in INNER_PREF if f != k and f in present)


def run_dir(out: Path) -> Path:
    """Big per-run files (weights, predictions) live in 6_Work/runs/<results folder name>."""
    return P.WORK / "runs" / out.name


def merge_rows(path: Path, new_rows: list, folds_done: set) -> pd.DataFrame:
    """Keep rows of folds that were not rerun (so --folds 2 3 does not erase fold 1)."""
    new = pd.DataFrame(new_rows)
    if path.exists():
        old = pd.read_csv(path)
        old = old[~old["fold"].isin(folds_done)]
        new = pd.concat([old, new], ignore_index=True)
    sort = ["fold", "epoch"] if "epoch" in new.columns else ["fold"]
    return new.sort_values(sort).reset_index(drop=True)


# ----------------------------------------------------------------------------- CV
def run(preset: str, out: Path, folds=None, max_minutes_per_fold=None, device=None):
    global DEVICE
    DEVICE = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = PRESETS[preset]
    out.mkdir(parents=True, exist_ok=True)
    rdir = run_dir(out); rdir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    cat = load_cat()
    data = load_all(cat)
    present = set(int(f) for f in cat.fold.unique())
    print(f"loaded {len(data)} chips; device {DEVICE}; torch threads {torch.get_num_threads()}", flush=True)
    print(f"tables -> {out}\nweights + predictions -> {rdir}", flush=True)
    variant = "b_no_haze"
    fits, hist = [], []
    done = set()
    for k in (folds or sorted(cat.fold.unique())):
        k = int(k)
        seed_all(k)
        t0 = time.time()
        held = cat[cat.fold == k]
        pool = cat[(cat.fold != k) & (~cat.hazy)]
        train = drop_near(pool, held)
        inner = pick_inner(k, present)
        iv = train[train.fold == inner]
        tr = drop_near(train[train.fold != inner], iv)
        tr_ids, iv_ids = list(tr.chip_id), list(iv.chip_id)
        mean, std = fold_stats(tr_ids, data)
        model = UNet(8, cfg["base"]).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
        rng = np.random.default_rng(k)
        best, best_ep, bad, best_state = -1, 0, 0, None
        for ep in range(1, cfg["epochs"] + 1):
            te = time.time(); model.train()
            ids = np.array(tr_ids * cfg["crops_per_chip"]); rng.shuffle(ids)
            losses = []
            for j in range(0, len(ids), cfg["batch"]):
                x, y, m = make_batch(ids[j:j + cfg["batch"]], data, mean, std, cfg["crop"], rng)
                x, y, m = x.to(DEVICE), y.to(DEVICE), m.to(DEVICE)
                opt.zero_grad(); l = loss_fn(model(x), y, m); l.backward(); opt.step(); losses.append(l.item())
            sched.step()
            probs = predict_prob(model, iv_ids, data, mean, std)
            viou = iou_at(probs, iv_ids, data, 0.5)
            hist.append({"fold": k, "epoch": ep, "train_loss": float(np.mean(losses)), "inner_val_iou@0.5": float(viou),
                         "lr": sched.get_last_lr()[0], "seconds": round(time.time() - te, 1)})
            print(f"fold {k} ep {ep:3d} loss {np.mean(losses):.4f} innerIoU {viou:.4f} {time.time()-te:.0f}s", flush=True)
            if viou > best:
                best, best_ep, bad = viou, ep, 0
                best_state = {n: v.clone() for n, v in model.state_dict().items()}
            else:
                bad += 1
            if bad >= cfg["patience"]:
                break
            if max_minutes_per_fold and (time.time() - t0) / 60 > max_minutes_per_fold:
                print(f"fold {k}: time cap reached at epoch {ep}", flush=True); break
        model.load_state_dict(best_state)
        torch.save({"state": {n: v.cpu() for n, v in best_state.items()}, "mean": mean, "std": std, "base": cfg["base"]},
                   rdir / f"unet_fold{k}.pt")
        probs = predict_prob(model, iv_ids, data, mean, std)
        grid = np.round(np.arange(0.05, 0.951, 0.05), 2)
        ious = [iou_at(probs, iv_ids, data, t) for t in grid]
        thr = float(grid[int(np.argmax(ious))])
        hp = predict_prob(model, list(held.chip_id), data, mean, std)
        for cid, p in hp.items():
            save_mask(rdir / "preds" / f"UNET_{variant}", cid, p > thr)
        rec = {"variant": variant, "fold": int(k), "held_site": ",".join(sorted(held.site.unique())),
               "held_chips": len(held), "train_chips": len(tr_ids), "inner_fold": inner, "inner_chips": len(iv_ids),
               "best_epoch": best_ep, "epochs_run": ep, "inner_iou@0.5": round(best, 4), "threshold": thr,
               "inner_iou@thr": round(float(max(ious)), 4), "minutes": round((time.time() - t0) / 60, 1)}
        fits.append(rec); print(rec, flush=True); done.add(int(k))
        merge_rows(out / "fold_fits.csv", fits, done).to_csv(out / "fold_fits.csv", index=False)
        merge_rows(out / "history.csv", hist, done).to_csv(out / "history.csv", index=False)
    json.dump({"preset": preset, **cfg, "device": str(DEVICE),
               "params": sum(p.numel() for p in UNet(8, cfg["base"]).parameters())},
              open(out / "config.json", "w"), indent=1)


def bench(preset):
    cfg = PRESETS[preset]
    cat = load_cat().head(40); data = load_all(cat); ids = list(cat.chip_id)
    mean, std = fold_stats(ids, data)
    model = UNet(8, cfg["base"]).to(DEVICE); print("params", sum(p.numel() for p in model.parameters()), "device", DEVICE)
    opt = torch.optim.Adam(model.parameters(), 1e-3); rng = np.random.default_rng(0)
    for s in range(6):
        t = time.time(); x, y, m = make_batch(rng.choice(ids, cfg["batch"]), data, mean, std, cfg["crop"], rng)
        x, y, m = x.to(DEVICE), y.to(DEVICE), m.to(DEVICE)
        t1 = time.time(); opt.zero_grad(); l = loss_fn(model(x), y, m); l.backward(); opt.step()
        print(f"step {s}: batch {t1-t:.2f}s train {time.time()-t1:.2f}s loss {l.item():.3f}")
    t = time.time(); predict_prob(model, ids[:16], data, mean, std); print(f"predict 16 full chips {time.time()-t:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="cpu", choices=list(PRESETS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--folds", type=int, nargs="*")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--crops-per-chip", type=int)
    ap.add_argument("--max-minutes-per-fold", type=float)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--device", choices=["cpu", "cuda"], help="default: cuda if available, else cpu")
    a = ap.parse_args()
    if a.epochs: PRESETS[a.preset]["epochs"] = a.epochs
    if a.crops_per_chip: PRESETS[a.preset]["crops_per_chip"] = a.crops_per_chip
    if a.bench:
        DEVICE = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        bench(a.preset)
    else:
        run(a.preset, Path(a.out) if a.out else P.unet_dir(a.preset), a.folds, a.max_minutes_per_fold, a.device)
