# LoRA fine-tuning for DINOv3

`train_dinov3_lora.py` fine-tunes DINOv3 with LoRA (low-rank adaptation) for mangrove segmentation.

In the frozen run (`train_dinov3.py --mode frozen`), DINOv3 is left exactly as Meta released it and only a small decoder is trained on top. LoRA also lets DINOv3 adapt to our task. Its original weights stay frozen, and each chosen layer gets a small trainable correction made of two thin matrices. With the default settings that is about 0.8 million trainable numbers, or 0.26% of the ViT-L model. The correction starts at zero, so before training the model behaves exactly like the frozen one.

## How it differs from the frozen run

Everything that affects the comparison is reused from `train_dinov3.py`: the Train/Val chips, the five leave-one-site-out folds, the haze rule and the one-chip buffer, the decoder, loss, optimiser and threshold choice, and the scoring through `evaluate.py`. Training length and batch size also match the frozen run (up to 40 epochs, patience 10, 8 chips per update). The output folder has the same files and columns, plus `lora_vs_frozen.csv`.

The differences are the ones LoRA needs. DINOv3 runs on every training step instead of being computed once and cached, so a GPU is needed. Flips are applied to the image before DINOv3 rather than to the cached features. On a GPU, DINOv3 runs in 16-bit precision with gradient checkpointing, which saves memory without changing the result. In `cv_results.csv` the `model` column says `DINOv3_LoRA`.

`train_dinov3.py` also has a simpler `--mode lora` with fixed settings. This script leaves it untouched and adds configurable settings, memory savers and the comparison table.

## Default settings

- Rank 8 (`--lora-r`). The LoRA paper found ranks of 4 to 8 enough for attention layers, and a small rank limits overfitting on our small folds.
- Alpha of twice the rank, so 16 (`--lora-alpha`). This keeps the correction equally strong if the rank is changed.
- Dropout 0.05 (`--lora-dropout`).
- The query and value projections of attention (`--lora-targets q_proj v_proj`). Other options are `k_proj`, `o_proj`, `up_proj` and `down_proj`.
- All transformer blocks. `--lora-last-n 12` adapts only the last 12.
- Learning rates of 0.001 for the decoder and 0.0001 for LoRA.

## How to run it

It needs the Train/Val dataset, the haze table (`QA_download.csv`) and a Hugging Face login with the DINOv3 licence accepted (see `HOW_TO_GET_DINOV3.md`). From `src/mangrove_segmentation`:

```
python dinov3/train_dinov3_lora.py --train-val <Train_Val_Dataset> --qa <QA_download.csv> --out <results folder> --frozen-results ../../results/mangrove_segmentation/dinov3
```

Add `--folds 1` for a quick first trial. A run of fewer than five folds gets no yes or no against the random forest in its `RESULTS.md`, because the random forest's 0.704 is a mean over all five folds. To measure GPU memory and speed without data or weights, run `python dinov3/train_dinov3_lora.py --bench`. The tests are in `tests/test_dinov3_lora.py` and `tests/test_dinov3_lora_verdict.py`.

The Meeting 6 minutes mention trying a higher input resolution. That is `--scale 4`. It is off by default because the frozen run used `--scale 2`, so a fair test would rerun the frozen model at 4 too.

## GPU memory

These numbers were measured on an 8 GB RTX 5060 Ti with a random-weight model the same size as DINOv3 ViT-L:

- Default settings: about 2.9 GB, roughly 18 seconds per epoch for a fold of about 120 training chips.
- `--no-grad-checkpoint --gpu-batch 4`: about 5.1 GB and roughly a quarter faster.
- `--scale 4`: about 6.1 GB and about six times slower.

The real five-fold run with the default settings used 2.9 GB at peak and about 13 seconds per epoch, for folds of 56 to 107 training chips. It took 36 minutes of training in total.

## Status

The script has been run on the real DINOv3 ViT-L weights and the real chips, on all five folds, with the default settings. The mean IoU is 0.694. The frozen run scored 0.686 and the random forest 0.704, so LoRA does not beat the random forest. The gain over the frozen run is 0.008, which is within noise: there is one seed per fold and the 95% intervals of the two runs overlap on every fold. Folds 2 and 3 ran all 40 epochs without stopping early. The default settings have not been tuned on our data.

The tables and the caveats are in [results/mangrove_segmentation/dinov3_lora/RESULTS.md](../../../results/mangrove_segmentation/dinov3_lora/RESULTS.md).
