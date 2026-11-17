"""DINOv3 with LoRA fine-tuning for mangrove segmentation, built to be compared with the frozen run.

WHAT THIS IS
  train_dinov3.py --mode frozen keeps DINOv3 exactly as Meta released it and trains only a small decoder.
  This script also lets DINOv3 itself adapt, through LoRA (low-rank adaptation): every chosen weight matrix W
  (frozen) gets a small trainable correction, W + (alpha / r) * B @ A, where A and B are two thin matrices of
  rank r. Only A, B and the decoder are trained; the 300 million original weights never change. B starts at
  zero, so the model starts out identical to the frozen one and can only move away from it if that helps.

WHY A SEPARATE FILE
  train_dinov3.py already has a basic `--mode lora` with fixed settings (r=8, alpha=16, q/v, all layers, no
  memory savers). This file does not change it. It imports train_dinov3.py and reuses its code for everything
  that must be the same as the frozen run, and adds: configurable rank / alpha / target layers, mixed precision,
  gradient accumulation and gradient checkpointing (so ViT-L fits on an 8 GB GPU), a LoRA-vs-frozen table, and
  a --bench mode that measures GPU memory and speed without needing the weights or the data.

IDENTICAL TO THE FROZEN RUN (taken from train_dinov3.py, not copied)
  * Inputs: Train_Val_Dataset, 2020 chips, label GMW v4.0.19; RGB = B04/B03/B02 stretched 2-98 % and upsampled
    x2 for DINOv3; the 8 spectral channels go to the decoder's conv branch. The Test_Dataset is never read.
  * Cross-validation: the 5 leave-one-site-out folds, hazy scenes dropped from training, the 1-chip buffer, and
    the same inner validation fold for early stopping and the decision threshold.
  * Decoder (FusionHead), loss, optimiser, schedule, seeds, augmentation, threshold grid.
  * Decoder starting weights: the decoder is created before the LoRA matrices, so for each fold it starts from
    the same random numbers as in the frozen run.
  * Training length: up to 40 epochs, patience 10, batch 8 (the frozen defaults; train_dinov3.py's own lora
    mode uses 30 / 8 / 4).
  * Scoring and outputs: evaluate.py through train_dinov3.score(); the same files and the same columns.

WHAT NECESSARILY DIFFERS
  * DINOv3 runs on every training step (the frozen run reads cached features), so it needs a GPU.
  * A flip augmentation flips the image before DINOv3 here; the frozen run flips the cached feature grid.
  * Gradient checkpointing is on by default. It gives exactly the same gradients, but keeps far less in GPU
    memory by recomputing each transformer block during the backward pass, so the full batch of 8 fits on an
    8 GB GPU. Measured on an RTX 5060 Ti (8 GB) for ViT-L with a random-weight model of the same size
    (--bench): batch 8 with checkpointing 2.9 GB and 0.13 s per chip; without it, 4 chips at a time take
    5.1 GB and 0.09 s per chip, and 8 at a time do not fit.
  * With --gpu-batch smaller than --batch, gradients are accumulated: the weights are updated once per --batch
    chips as in the frozen run, but the decoder's BatchNorm sees --gpu-batch chips at a time. Not needed with
    the defaults.
  * With mixed precision (default on a GPU) DINOv3 runs in 16-bit floats. The frozen run's cached features are
    also stored as 16-bit floats. The decoder and the loss stay in 32-bit in both.

EXAMPLES (from the SDP folder, 'mangrove' environment active)
  python 2_Code/dinov3/train_dinov3_lora.py                                  # defaults: r=8, alpha=16, q+v, all layers
  python 2_Code/dinov3/train_dinov3_lora.py --no-grad-checkpoint --gpu-batch 4   # faster, more GPU memory (see below)
  python 2_Code/dinov3/train_dinov3_lora.py --lora-r 16 --lora-alpha 32      # more capacity
  python 2_Code/dinov3/train_dinov3_lora.py --lora-targets q_proj k_proj v_proj o_proj --lora-last-n 12
  python 2_Code/dinov3/train_dinov3_lora.py --bench --bench-arch vitl                  # memory + speed, no data needed
  python 2_Code/dinov3/train_dinov3_lora.py --smoke --force --cpu --out <temp folder>  # tiny random model: code test
Needs the Hugging Face login with the DINOv3 licence accepted (HOW_TO_GET_DINOV3.md), except --smoke and --bench.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_dinov3 as T  # noqa: E402  the frozen pipeline: data rules, decoder, loss, scoring

# ---------------------------------------------------------------------------------------------- LoRA defaults
# rank r = 8        Size of the two thin matrices. The LoRA paper found ranks of 4-8 enough to adapt attention
#                   layers. Each fold has only 56-107 training chips, so a small rank also limits overfitting.
#                   For ViT-L with the defaults below this adds about 0.8 million trainable numbers to a model of
#                   about 300 million (about 0.3 %).
# alpha = 2 * r     The correction is multiplied by alpha / r. Keeping alpha = 2r (16 for r = 8) is the common
#                   convention: the correction has the same strength whatever rank is tried, so changing r does
#                   not silently change the effective learning rate. It is what train_dinov3.py's lora mode uses.
# dropout = 0.05    Light regularisation on the LoRA path only (same as train_dinov3.py).
# targets = q, v    The query and value projections of self-attention. The LoRA paper found adapting these two
#                   gives the best result per added parameter. k_proj, o_proj (the rest of attention) and
#                   up_proj, down_proj (the MLP) can be added for more capacity at more memory and time.
# layers = all      Every transformer block gets adapters. --lora-last-n N adapts only the last N blocks, which
#                   are the most task-specific ones.
#                   Note: this does NOT save GPU memory or time by itself (DINOv3 still runs in full and the
#                   early blocks' activations are still kept); it only reduces what can change.
# learning rates    Decoder 1e-3 (the frozen value), LoRA 1e-4 (train_dinov3.py's value): the pretrained features
#                   should move slowly while the new decoder learns quickly.
LORA_R, LORA_DROPOUT = 8, 0.05
LORA_TARGETS = ["q_proj", "v_proj"]
LR_HEAD, LR_LORA = 1e-3, 1e-4
EPOCHS, PATIENCE, BATCH = (T.DEFAULTS["frozen"][k] for k in ("epochs", "patience", "batch"))   # 40, 10, 8

MODEL_LABEL = "DINOv3_LoRA"          # value of the `model` column in cv_results.csv (the frozen run writes "DINOv3")
BENCH_ARCH = {                       # published DINOv3 ViT sizes, for --bench only (random weights)
    "vits": dict(hidden_size=384, num_hidden_layers=12, num_attention_heads=6, intermediate_size=1536),
    "vitb": dict(hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072),
    "vitl": dict(hidden_size=1024, num_hidden_layers=24, num_attention_heads=16, intermediate_size=4096),
}


# ============================================================================ LoRA
def layer_index(name: str):
    """'layer.11.attention.q_proj' -> 11 (the transformer block a module belongs to), or None."""
    m = re.search(r"(?:^|\.)(\d+)\.", name)
    return int(m.group(1)) if m else None


def resolve_targets(backbone, targets, last_n=None) -> list:
    """Full names of the linear layers that get a LoRA adapter: name ends with one of `targets`, and (optionally)
    the layer sits in one of the last `last_n` transformer blocks."""
    linear = [n for n, m in backbone.named_modules() if isinstance(m, torch.nn.Linear)]
    hits = [n for n in linear if n.split(".")[-1] in set(targets) and layer_index(n) is not None]
    missing = sorted(set(targets) - {n.split(".")[-1] for n in hits})
    if missing or not hits:
        raise SystemExit(f"--lora-targets {missing or list(targets)} not found in the model. Linear layers inside the "
                         f"transformer blocks are named: {sorted({n.split('.')[-1] for n in linear if layer_index(n) is not None})}")
    if last_n:
        n_blocks = max(layer_index(n) for n in hits) + 1
        if not 0 < last_n <= n_blocks:
            raise SystemExit(f"--lora-last-n must be between 1 and {n_blocks} (the model has {n_blocks} blocks)")
        hits = [n for n in hits if layer_index(n) >= n_blocks - last_n]
    return hits


def add_lora(backbone, r=LORA_R, alpha=None, dropout=LORA_DROPOUT, targets=None, last_n=None, grad_checkpoint=False):
    """Wrap a DINOv3 backbone with LoRA adapters (peft). Returns (model, info). All original weights are frozen.
    grad_checkpoint trades speed for GPU memory during training; it does not change the result."""
    from peft import LoraConfig, get_peft_model
    alpha = 2 * r if alpha is None else alpha
    names = resolve_targets(backbone, targets or LORA_TARGETS, last_n)
    if grad_checkpoint:
        if not getattr(backbone, "supports_gradient_checkpointing", False):
            raise SystemExit("this model does not support gradient checkpointing; add --no-grad-checkpoint")
        # transformers 5 renamed this switch; older versions (>= 4.56) only have the second name
        (getattr(backbone, "activation_checkpointing_enable", None) or backbone.gradient_checkpointing_enable)()
    model = get_peft_model(backbone, LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=names,
                                                bias="none"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    info = {"lora_r": r, "lora_alpha": alpha, "lora_scaling": alpha / r, "lora_dropout": dropout,
            "lora_targets": sorted({n.split(".")[-1] for n in names}), "lora_last_n": last_n,
            "lora_modules": len(names), "lora_blocks": len({layer_index(n) for n in names}),
            "trainable_backbone_params": int(trainable), "backbone_params": int(total),
            "trainable_pct": round(100 * trainable / total, 4)}
    return model, info


def lora_state(model) -> dict:
    return {n: v.detach().clone() for n, v in model.state_dict().items() if "lora_" in n}


def describe(info: dict) -> str:
    where = "all blocks" if not info["lora_last_n"] else f"the last {info['lora_last_n']} blocks"
    return (f"LoRA r={info['lora_r']}, alpha={info['lora_alpha']} on {', '.join(info['lora_targets'])} in {where} "
            f"({info['lora_modules']} layers, {info['trainable_backbone_params']:,} trainable = {info['trainable_pct']} % "
            f"of the backbone)")


# ============================================================================ training helpers
def autocast(device: str, amp: bool):
    """16-bit floats for the DINOv3 forward pass on a GPU (about half the memory, faster). Off on a CPU."""
    return torch.autocast("cuda", dtype=torch.float16) if (amp and device == "cuda") else nullcontext()


@torch.no_grad()
def predict(run_, head, lora_bb, ids, mean, std, amp, bs=4) -> dict:
    """As train_dinov3.Runner.predict, with the backbone under autocast and the decoder in 32-bit."""
    head.eval(); lora_bb.eval()
    out = {}
    for j in range(0, len(ids), bs):
        b = ids[j:j + bs]
        with autocast(run_.device, amp):
            tok, spec, _, _ = run_.batch(b, mean, std, None, False, lora_bb)
        p = torch.sigmoid(head(tok.float(), spec, spec.shape[-2:])).float().cpu().numpy()
        out.update(dict(zip(b, p)))
    return out


def train_epoch(run_, head, lora_bb, opt, scaler, tr_ids, mean, std, rng, batch, gpu_batch, amp) -> float:
    """One pass over the training chips. The weights are updated once per `batch` chips; `gpu_batch` chips go
    through the GPU at a time (gradient accumulation). With gpu_batch == batch this is train_dinov3.py's loop."""
    head.train(); lora_bb.train()
    ids = np.array(tr_ids); rng.shuffle(ids)
    losses = []
    for j in range(0, len(ids), batch):
        group = ids[j:j + batch]
        opt.zero_grad(set_to_none=True)
        for i in range(0, len(group), gpu_batch):
            part = group[i:i + gpu_batch]
            with autocast(run_.device, amp):
                tok, spec, y, m = run_.batch(part, mean, std, rng, True, lora_bb)
            loss = T.loss_fn(head(tok.float(), spec, y.shape[-2:]), y, m)
            scaler.scale(loss * len(part) / len(group)).backward()     # chip-weighted mean over the group
            losses.append(loss.item())
        scaler.step(opt); scaler.update()
    return float(np.mean(losses))


def default_run_name(a, info) -> str:
    name = f"{'SMOKE_' if a.smoke else ''}lora_{'smoke' if a.smoke else a.model_id.split('/')[-1]}"
    if (info["lora_r"], info["lora_alpha"], info["lora_targets"], info["lora_last_n"]) != (LORA_R, 2 * LORA_R,
                                                                                        sorted(LORA_TARGETS), None):
        name += f"_r{info['lora_r']}_a{info['lora_alpha']:g}_" + "-".join(t.replace("_proj", "") for t in info["lora_targets"])
        name += f"_last{info['lora_last_n']}" if info["lora_last_n"] else ""
    return name + ("_nospec" if a.no_spectral else "")


# ============================================================================ cross-validation
def run(a) -> Path:
    device = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
    if device == "cpu" and not a.force:
        raise SystemExit("LoRA fine-tuning on a CPU would take days. Use an NVIDIA GPU (or Colab), or run "
                         "train_dinov3.py --mode frozen. Add --force only to test the code path.")
    a.mode = "lora"                                   # read by train_dinov3.score()
    amp = (device == "cuda") if a.amp is None else (a.amp and device == "cuda")
    gpu_batch = min(a.gpu_batch or a.batch, a.batch)
    if a.qa:                                          # haze table given explicitly instead of found by dino_paths
        T.DP.qa_csv = lambda: Path(a.qa)
    tv = Path(a.train_val) if a.train_val else T.DP.TRAIN_VAL
    if not (tv / "catalog.csv").exists():
        raise SystemExit(f"Train/Val dataset not found: {tv}\nPass --train-val <Train_Val_Dataset folder> (and --qa "
                         "<QA_download.csv>), or build it first with run_all.py --step datasets.")

    cat = T.load_cat(tv)
    all_folds = sorted(int(k) for k in cat.fold.unique())
    folds = [int(k) for k in (a.folds or all_folds)]
    if a.max_chips:   # smoke/dry-run: keep a few chips per fold
        cat = pd.concat([g.sample(min(len(g), a.max_chips), random_state=0) for _, g in cat.groupby("fold")],
                        ignore_index=True)
    data = T.load_all(cat, tv)
    run_ = T.Runner(a, data, device)                  # reads patch size, feature width and the RGB mean/std
    _, info = add_lora(run_.backbone, a.lora_r, a.lora_alpha, a.lora_dropout, a.lora_targets, a.lora_last_n)
    run_.backbone = None                              # each fold loads its own fresh copy; don't keep two on the GPU
    if device == "cuda":
        torch.cuda.empty_cache()

    name = a.run_name or default_run_name(a, info)
    out = Path(a.out) / name if a.out else T.DP.DINO_RESULTS / name
    out.mkdir(parents=True, exist_ok=True)
    print(f"device {device} | LoRA | model {'random tiny (smoke)' if a.smoke else a.model_id}\n{describe(info)}\n"
          f"batch {a.batch} ({gpu_batch} on the GPU at a time) | mixed precision {amp} | "
          f"gradient checkpointing {a.grad_checkpoint}\nTrain/Val: {tv}\nresults : {out}", flush=True)

    fits, hist, extra = [], [], []
    for k in folds:
        T.seed_all(k)
        t0 = time.time()
        held = cat[cat.fold == k]
        train = T.drop_near(cat[(cat.fold != k) & (~cat.hazy)], held)
        inner = next(f for f in T.INNER_PREF if f != k and f in set(train.fold))
        iv = train[train.fold == inner]
        tr = T.drop_near(train[train.fold != inner], iv)
        if tr.empty:
            tr = train[train.fold != inner]
        tr_ids, iv_ids = list(tr.chip_id), list(iv.chip_id)
        mean, std = T.fold_stats(tr_ids, data)
        # decoder first, so that it starts from the same random numbers as in the frozen run of this fold
        head = T.FusionHead(run_.c_tok, spectral=not a.no_spectral).to(device)
        bb, *_ = T.load_backbone(a.model_id, a.smoke, device)             # fresh pretrained weights for every fold
        lora_bb, _ = add_lora(bb, a.lora_r, a.lora_alpha, a.lora_dropout, a.lora_targets, a.lora_last_n,
                              a.grad_checkpoint)
        lora_bb.to(device)
        opt = torch.optim.AdamW([{"params": head.parameters(), "lr": a.lr_head},
                                 {"params": [p for p in lora_bb.parameters() if p.requires_grad], "lr": a.lr_lora}],
                                weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        rng = np.random.default_rng(k)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        best, best_ep, bad, best_state, ep = -1.0, 0, 0, None, 0
        for ep in range(1, a.epochs + 1):
            te = time.time()
            loss = train_epoch(run_, head, lora_bb, opt, scaler, tr_ids, mean, std, rng, a.batch, gpu_batch, amp)
            sched.step()
            probs = predict(run_, head, lora_bb, iv_ids, mean, std, amp)
            viou = T.iou_at(probs, iv_ids, data, 0.5)
            hist.append({"fold": k, "epoch": ep, "train_loss": loss, "inner_val_iou@0.5": float(viou),
                         "seconds": round(time.time() - te, 1)})
            print(f"fold {k} ep {ep:3d} loss {loss:.4f} innerIoU {viou:.4f} {time.time()-te:.0f}s", flush=True)
            if viou > best:
                best, best_ep, bad = viou, ep, 0
                best_state = {"head": {n: v.detach().clone() for n, v in head.state_dict().items()},
                              "lora": lora_state(lora_bb)}
            else:
                bad += 1
            if bad >= a.patience:
                break
        head.load_state_dict(best_state["head"])
        lora_bb.load_state_dict(best_state["lora"], strict=False)
        torch.save({"head": best_state["head"], "lora": best_state["lora"], "mean": mean, "std": std,
                    "model_id": a.model_id, "spectral": not a.no_spectral, "lora_config": info},
                   out / f"dinov3_fold{k}.pt")
        probs = predict(run_, head, lora_bb, iv_ids, mean, std, amp)
        grid = np.round(np.arange(0.05, 0.951, 0.05), 2)
        ious = [T.iou_at(probs, iv_ids, data, t) for t in grid]
        thr = float(grid[int(np.argmax(ious))])
        hp = predict(run_, head, lora_bb, list(held.chip_id), mean, std, amp)
        for cid, p in hp.items():
            T.save_mask(out / "preds" / f"DINO_{T.VARIANT}", cid, p > thr)
        rec = {"variant": T.VARIANT, "fold": k, "held_site": ",".join(sorted(held.site.unique())),
               "held_chips": len(held), "train_chips": len(tr_ids), "inner_fold": inner, "inner_chips": len(iv_ids),
               "best_epoch": best_ep, "epochs_run": ep, "inner_iou@0.5": round(best, 4), "threshold": thr,
               "inner_iou@thr": round(float(max(ious)), 4), "minutes": round((time.time() - t0) / 60, 1)}
        fits.append(rec); print(rec, flush=True)
        extra.append({"fold": k, "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20) if device == "cuda" else None,
                      "seconds_per_epoch": round(float(np.mean([h["seconds"] for h in hist if h["fold"] == k])), 1)})
        pd.DataFrame(fits).to_csv(out / "fold_fits.csv", index=False)
        pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
        del lora_bb, bb, head, opt
        if device == "cuda":
            torch.cuda.empty_cache()
    # config.json: the keys train_dinov3.score() reads, then the LoRA-only ones
    json.dump({"mode": "lora", "model_id": "random tiny (smoke)" if a.smoke else a.model_id, "smoke": a.smoke,
               "spectral_branch": not a.no_spectral, "scale": a.scale, "last4": a.last4, "aug": a.aug,
               "lora_r": a.lora_r, "device": device, "epochs": a.epochs, "patience": a.patience, "batch": a.batch,
               "lr_head": a.lr_head, "lr_lora": a.lr_lora, **info, "gpu_batch": gpu_batch, "mixed_precision": amp,
               "grad_checkpoint": a.grad_checkpoint,
               "per_fold": extra}, open(out / "config.json", "w"), indent=1)
    T.score(out, tv, cat, a)                          # evaluate.py scoring, figures and RESULTS.md, as for frozen
    finish_report(out, a, info, gpu_batch, amp, extra, folds, all_folds)
    return out


# ============================================================================ report additions
def frozen_cv_path(a) -> Path:
    if a.frozen_results:
        p = Path(a.frozen_results)
        return p / "cv_results.csv" if p.is_dir() else p
    tag = f"{'SMOKE_' if a.smoke else ''}frozen_{'smoke' if a.smoke else a.model_id.split('/')[-1]}"
    return (Path(a.out) if a.out else T.DP.DINO_RESULTS) / f"{tag}{'_nospec' if a.no_spectral else ''}" / "cv_results.csv"


def verdict_line(mean_iou: float, folds, all_folds, smoke: bool) -> str:
    """The sentence after '**Verdict:**' in RESULTS.md.

    The random-forest bar (T.BAR_B2) is a mean over ALL folds. It is only compared with a run that also covers all
    folds. A partial run (--folds) gets no yes/no: one fold against a 5-fold mean says nothing, because some sites
    are much easier than others."""
    if smoke:
        return "SMOKE TEST: random weights, the numbers mean nothing. This run only proves the code works."
    n = len(set(all_folds))
    if set(folds) != set(all_folds):
        done = sorted(set(folds))
        return (f"**No verdict: partial run** (fold{'s' if len(done) > 1 else ''} {', '.join(map(str, done))} of {n}). "
                f"The random-forest bar of {T.BAR_B2} is a mean over all {n} folds, so it is not compared with this "
                f"run's {mean_iou:.3f}. Run all folds for a verdict.")
    answer = "**Yes: beats the random forest**" if mean_iou > T.BAR_B2 else "**No: does not beat the random forest yet**"
    return f"{answer} (DINOv3 LoRA {mean_iou:.3f} vs B2 {T.BAR_B2}, both means over all {n} folds)"


def set_verdict(md: str, line: str) -> str:
    """Replace the verdict that train_dinov3.score() wrote (it always compares with the all-fold mean)."""
    return re.sub(r"^\*\*Verdict:\*\* .*$", lambda _: f"**Verdict:** {line}", md, count=1, flags=re.M)


def finish_report(out: Path, a, info: dict, gpu_batch: int, amp: bool, extra: list, folds, all_folds) -> None:
    """Label the tables as LoRA, state the real LoRA settings in RESULTS.md, correct the verdict for partial runs,
    and add the LoRA-vs-frozen table. `folds` are the folds of this run, `all_folds` every fold in the catalog."""
    for f in ("cv_results.csv", "per_chip.csv"):
        df = pd.read_csv(out / f); df["model"] = MODEL_LABEL; df.to_csv(out / f, index=False)
    cv = pd.read_csv(out / "cv_results.csv").set_index("fold")
    md = (out / "RESULTS.md").read_text(encoding="utf-8")
    md = md.replace(f"LoRA r={a.lora_r} on attention q/v", describe(info))
    md = set_verdict(md, verdict_line(float(cv["iou"].mean()), folds, all_folds, a.smoke))
    peak = [e["peak_gpu_mb"] for e in extra if e["peak_gpu_mb"]]
    md += f"""
## LoRA settings

- **Adapters:** {describe(info)}. Correction strength alpha / r = {info['lora_scaling']:g}, dropout {info['lora_dropout']}.
- **Learning rates:** decoder {a.lr_head:g}, LoRA {a.lr_lora:g}.
- **Batch:** {a.batch} chips per weight update, {gpu_batch} on the GPU at a time{' (gradients accumulated)' if gpu_batch < a.batch else ''};
  mixed precision {'on' if amp else 'off'}; gradient checkpointing {'on' if a.grad_checkpoint else 'off'}.
- **Measured:** {f'peak GPU memory {max(peak):,} MB; ' if peak else ''}{np.mean([e['seconds_per_epoch'] for e in extra]):.1f} s per epoch on average.
"""
    fz = frozen_cv_path(a)
    if fz.exists():
        f = pd.read_csv(fz).set_index("fold")
        both = cv.index.intersection(f.index)
        rows = []
        for k in both:
            overlap = not (cv.loc[k, "iou_ci95_lo"] > f.loc[k, "iou_ci95_hi"] or f.loc[k, "iou_ci95_lo"] > cv.loc[k, "iou_ci95_hi"])
            rows.append({"Fold": k, "Held-out site": str(cv.loc[k, "held_site"]).replace("_", " "),
                         "Frozen IoU (95% CI)": f"{f.loc[k,'iou']:.3f} ({f.loc[k,'iou_ci95_lo']:.3f}–{f.loc[k,'iou_ci95_hi']:.3f})",
                         "LoRA IoU (95% CI)": f"{cv.loc[k,'iou']:.3f} ({cv.loc[k,'iou_ci95_lo']:.3f}–{cv.loc[k,'iou_ci95_hi']:.3f})",
                         "LoRA − frozen": f"{cv.loc[k,'iou'] - f.loc[k,'iou']:+.3f}", "CIs overlap": "yes" if overlap else "no"})
        tab = pd.DataFrame(rows)
        tab.to_csv(out / "lora_vs_frozen.csv", index=False)
        md += f"""
## LoRA against the frozen run (same folds, same scoring)

Mean IoU over {len(both)} fold(s): frozen {f.loc[both, 'iou'].mean():.3f}, LoRA {cv.loc[both, 'iou'].mean():.3f}
(difference {cv.loc[both, 'iou'].mean() - f.loc[both, 'iou'].mean():+.3f}). One seed per fold: differences under about 0.03 are within noise.
Frozen results read from `{fz}`.

{tab.to_markdown(index=False)}
"""
    else:
        md += f"\nNo frozen run found at `{fz}`, so there is no LoRA-vs-frozen table. Pass --frozen-results <folder>.\n"
    (out / "RESULTS.md").write_text(md, encoding="utf-8")


# ============================================================================ benchmark (no weights, no data)
def bench(a) -> dict:
    """Peak GPU memory and seconds per training step for a random-weight DINOv3 of the chosen size, with the same
    chip size, decoder, loss and LoRA settings as a real run. Memory and speed do not depend on the weight values."""
    from transformers import DINOv3ViTConfig, DINOv3ViTModel
    device = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
    amp = (device == "cuda") if a.amp is None else (a.amp and device == "cuda")
    gpu_batch = min(a.gpu_batch or a.batch, a.batch)
    px = a.bench_chip * a.scale
    bb = DINOv3ViTModel(DINOv3ViTConfig(patch_size=16, image_size=px, num_register_tokens=4, **BENCH_ARCH[a.bench_arch]))
    lora_bb, info = add_lora(bb, a.lora_r, a.lora_alpha, a.lora_dropout, a.lora_targets, a.lora_last_n, a.grad_checkpoint)
    lora_bb.to(device).train()
    head = T.FusionHead(bb.config.hidden_size, spectral=not a.no_spectral).to(device).train()
    opt = torch.optim.AdamW([{"params": head.parameters(), "lr": a.lr_head},
                             {"params": [p for p in lora_bb.parameters() if p.requires_grad], "lr": a.lr_lora}])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    g = torch.Generator().manual_seed(0)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    times = []
    for step in range(a.bench_steps + 1):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        for i in range(0, a.batch, gpu_batch):
            n = min(gpu_batch, a.batch - i)
            x = torch.randn(n, 3, px, px, generator=g).to(device)
            spec = torch.randn(n, 8, a.bench_chip, a.bench_chip, generator=g).to(device)
            y = (torch.rand(n, a.bench_chip, a.bench_chip, generator=g) > 0.9).float().to(device)
            with autocast(device, amp):
                tok = T.patch_tokens(lora_bb, x, 16, a.last4)
            loss = T.loss_fn(head(tok.float(), spec, y.shape[-2:]), y, torch.ones_like(y))
            scaler.scale(loss * n / a.batch).backward()
        scaler.step(opt); scaler.update()
        if device == "cuda":
            torch.cuda.synchronize()
        if step:                                       # the first step includes one-off start-up cost
            times.append(time.time() - t0)
    res = {"arch": a.bench_arch, "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
           "image_px": px, "tokens": (px // 16) ** 2, "batch": a.batch, "gpu_batch": gpu_batch, "mixed_precision": amp,
           "grad_checkpoint": bool(a.grad_checkpoint), **info,
           "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20) if device == "cuda" else None,
           "seconds_per_update": round(float(np.mean(times)), 3),
           "seconds_per_chip": round(float(np.mean(times)) / a.batch, 4)}
    print(json.dumps(res, indent=1))
    return res


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", default=T.DEFAULT_MODEL, help=f"default {T.DEFAULT_MODEL}; small: {T.SMALL_MODEL}")
    # ---- LoRA
    ap.add_argument("--lora-r", type=int, default=LORA_R, help=f"rank of the adapters (default {LORA_R})")
    ap.add_argument("--lora-alpha", type=float, default=None, help="scaling numerator (default 2 x rank)")
    ap.add_argument("--lora-dropout", type=float, default=LORA_DROPOUT)
    ap.add_argument("--lora-targets", nargs="+", default=LORA_TARGETS,
                    help="which linear layers get adapters: q_proj k_proj v_proj o_proj (attention), up_proj down_proj (MLP)")
    ap.add_argument("--lora-last-n", type=int, default=None, help="only the last N transformer blocks (default: all)")
    ap.add_argument("--lr-lora", type=float, default=LR_LORA); ap.add_argument("--lr-head", type=float, default=LR_HEAD)
    # ---- training length and memory
    ap.add_argument("--epochs", type=int, default=EPOCHS); ap.add_argument("--patience", type=int, default=PATIENCE)
    ap.add_argument("--batch", type=int, default=BATCH, help=f"chips per weight update (default {BATCH}, as frozen)")
    ap.add_argument("--gpu-batch", type=int, default=None,
                    help="chips on the GPU at a time (default: --batch). Lower it if the GPU runs out of memory")
    ap.add_argument("--amp", dest="amp", action="store_true", default=None, help="mixed precision (default on a GPU)")
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--no-grad-checkpoint", dest="grad_checkpoint", action="store_false",
                    help="switch gradient checkpointing off: about a quarter less time per epoch, but much more GPU "
                         "memory (on an 8 GB GPU add --gpu-batch 4)")
    # ---- same switches as train_dinov3.py
    ap.add_argument("--folds", type=int, nargs="*")
    ap.add_argument("--scale", type=int, default=2, help="upsampling of the 256 px chip before DINOv3 (2 -> 32x32 tokens)")
    ap.add_argument("--last4", action="store_true", help="average the last 4 layers instead of the last one")
    ap.add_argument("--no-spectral", action="store_true", help="ablation: DINOv3 RGB features only")
    ap.add_argument("--aug", choices=["none", "flip"], default="flip")
    ap.add_argument("--train-val", default=None, help="default 6_Work/datasets/Train_Val_Dataset")
    ap.add_argument("--qa", default=None, help="haze table (site, date, water_b04_p5); default: found by dino_paths")
    ap.add_argument("--out", default=None, help="parent folder of the run; default 4_Results/DINOv3")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--frozen-results", default=None, help="folder of the frozen run to compare with (default: next to this run)")
    ap.add_argument("--max-chips", type=int, default=None, help="keep only N chips per fold (quick tests)")
    ap.add_argument("--cpu", action="store_true", help="force CPU even if a GPU is available")
    ap.add_argument("--force", action="store_true", help="allow running on a CPU (code test only)")
    ap.add_argument("--smoke", action="store_true", help="tiny random DINOv3 instead of the real weights (tests code only)")
    # ---- benchmark
    ap.add_argument("--bench", action="store_true", help="measure GPU memory and speed with a random model; no data needed")
    ap.add_argument("--bench-arch", choices=list(BENCH_ARCH), default="vitl")
    ap.add_argument("--bench-steps", type=int, default=5); ap.add_argument("--bench-chip", type=int, default=256)
    a = ap.parse_args(argv)
    if a.smoke and not a.out:
        ap.error("--smoke needs --out <temp folder>, so smoke results never land in 4_Results")
    if a.lora_r < 1 or a.batch < 1 or (a.gpu_batch is not None and a.gpu_batch < 1):
        ap.error("--lora-r, --batch and --gpu-batch must be at least 1")
    return a


if __name__ == "__main__":
    args = parse()
    bench(args) if args.bench else run(args)
