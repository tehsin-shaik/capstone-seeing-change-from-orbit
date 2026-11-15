# LoRA fine-tuning for DINOv3: what was added

16 Nov 2026, branch `zain-dinov3-lora`. Three new files. No existing file was edited, and nothing has been committed or pushed.

| File | What it is |
| --- | --- |
| [src/mangrove_segmentation/dinov3/train_dinov3_lora.py](../src/mangrove_segmentation/dinov3/train_dinov3_lora.py) | Trains DINOv3 with LoRA, set up to be compared fairly with the frozen run |
| [src/mangrove_segmentation/tests/test_dinov3_lora.py](../src/mangrove_segmentation/tests/test_dinov3_lora.py) | 10 automatic tests, on a tiny random model and made-up chips |
| [docs/CHANGES_ZAIN_LORA.md](CHANGES_ZAIN_LORA.md) | This summary |

The short version:

- The LoRA script is written and tested. All 20 tests in the repository pass (10 old, 10 new).
- **It has not been run on the real DINOv3 weights or the real mangrove chips, so there is no LoRA score yet.** Neither the weights nor the chips are on this PC.
- This PC can run it. It has an NVIDIA RTX 5060 Ti with 8 GB of memory, and PyTorch can use it. With the default settings, LoRA on the large DINOv3 model needs about 2.9 GB of GPU memory, measured on this PC with a random-weight model of the same size.

## 1. What LoRA is

DINOv3 is a large image model that Meta trained on hundreds of millions of images. The project already uses it "frozen": DINOv3 is left exactly as Meta released it, and only a small decoder on top is trained to turn its output into a mangrove map. That scored IoU 0.686, just under the random forest (0.704).

Fine-tuning means letting DINOv3 itself adjust to our task. Doing that to all of its roughly 300 million numbers needs a lot of GPU memory and, with only 56 to 107 training chips per fold, would quickly memorise the training chips instead of learning.

LoRA ("low-rank adaptation") is a cheap way to fine-tune. The original numbers stay frozen. Next to each chosen layer, LoRA adds a small correction built from two thin tables of numbers, and only those are trained. Three properties matter for us:

- **Small.** With the default settings it trains 786,432 numbers, which is 0.26% of the model.
- **Starts from the frozen model.** The correction starts at exactly zero, so before training the LoRA model gives the same output as the frozen one. It moves away only if that reduces the training error. (A test checks this.)
- **Fits on a normal GPU.** See section 5.

## 2. Something to know first: the repository already had a basic LoRA mode

`train_dinov3.py` already accepts `--mode lora`. The repository has no results from it, and the settings are fixed in the code: rank 8, strength 16, the same two layer types, all blocks. Only the rank can be changed. It also has no memory savers, so on the large model it probably does not fit on an 8 GB GPU (not measured).

The new script does not replace or change it. It imports `train_dinov3.py` and reuses its code, and adds:

- LoRA rank, strength (alpha), dropout, layer types and number of blocks as command-line options.
- The frozen run's training length and batch size, where the old LoRA mode used shorter training and a smaller batch.
- Three memory savers, so the large model fits on an 8 GB GPU.
- A LoRA-against-frozen table in the results.
- A benchmark mode that measures GPU memory and speed without needing the weights or the data.

## 3. What changed compared with the frozen version

**The same as the frozen run.** These come from `train_dinov3.py` itself. The new script calls that code, so the two cannot drift apart.

| | Frozen and LoRA both use |
| --- | --- |
| Data | Train/Val dataset only, 2020 chips, label GMW v4.0.19. The locked test set is never read |
| What DINOv3 sees | Red, green, blue, stretched per chip and enlarged 2× |
| What the decoder sees | DINOv3's output plus all 8 channels (6 bands, NDVI, MNDWI) |
| Cross-validation | The same 5 leave-one-site-out folds, hazy scenes dropped from training, the one-chip buffer, the same inner fold for early stopping and for the decision threshold |
| Decoder, loss, optimiser, schedule, seeds, augmentation | Identical. The decoder also starts from the same random numbers in each fold |
| Training length | Up to 40 epochs, stop after 10 epochs without improvement, 8 chips per weight update |
| Scoring | `evaluate.py`, through the frozen script's own scoring function |
| Output | The same files with the same columns. One extra file: `lora_vs_frozen.csv` |

**What is different, and why.**

| Difference | Why |
| --- | --- |
| DINOv3 gets LoRA adapters and is trained | That is the experiment |
| DINOv3 runs on every training step. The frozen run computes its output once and saves it | Its output now changes during training. This is why LoRA needs a GPU |
| Flips are applied to the image before DINOv3. The frozen run flips the saved output | Follows from the line above |
| DINOv3 runs in 16-bit numbers on the GPU ("mixed precision") | Halves the time. The frozen run's saved output is also 16-bit. The decoder and the loss stay 32-bit in both |
| Gradient checkpointing is on | Gives the same result with far less GPU memory, at about a third more time per epoch. It is what lets 8 chips go through an 8 GB GPU at once |
| The `model` column in `cv_results.csv` says `DINOv3_LoRA` | So that LoRA and frozen rows can be told apart when tables are joined. The frozen run writes `DINOv3` |
| `RESULTS.md` has two extra sections | The LoRA settings, and the LoRA-against-frozen table |

**The default LoRA settings.** The reasons are also in comments at the top of the script.

| Setting | Default | Reason |
| --- | --- | --- |
| Rank `--lora-r` | 8 | The LoRA paper found ranks of 4 to 8 enough for attention layers. With so few training chips, a small rank also limits memorising |
| Strength `--lora-alpha` | 2 × rank (16) | The usual convention. It keeps the correction equally strong whatever rank is tried |
| Dropout `--lora-dropout` | 0.05 | Light protection against memorising |
| Layer types `--lora-targets` | `q_proj v_proj` | The "query" and "value" parts of attention. The LoRA paper found these give the most improvement per added number. Others available: `k_proj`, `o_proj`, `up_proj`, `down_proj` |
| Blocks `--lora-last-n` | all 24 | `--lora-last-n 12` would adapt only the last 12. This limits what can change. It does not save memory or time |
| Learning rates | decoder 0.001, LoRA 0.0001 | The pretrained model should move slowly while the new decoder learns quickly. Same values as the existing code |

Rank 8, strength 16, dropout 0.05 and the two layer types are the same values as the existing `--mode lora`.

## 4. How to run it

All commands are typed in the **Anaconda Prompt** (Start menu → "Anaconda Prompt"), after `conda activate mangrove`. Always activate first: running the environment's Python without activating it crashes on this PC.

**The tests (work today):**

```
conda activate mangrove
cd C:\Users\Zain\Documents\capstone-seeing-change-from-orbit\src\mangrove_segmentation
python -m pytest -q tests/
```

**The speed and memory benchmark (works today, needs no data or weights):**

```
python dinov3\train_dinov3_lora.py --bench --bench-arch vitl
```

**The real run.** It needs three things that are not on this PC yet:

1. **The Train/Val dataset** (`6_Work/datasets/Train_Val_Dataset`) and the haze table (`QA_download.csv`), from the team's shared storage.
2. **The DINOv3 weights.** They are free but gated: accept Meta's licence on the model page and log in once with `huggingface-cli login`. The steps are in [HOW_TO_GET_DINOV3.md](../src/mangrove_segmentation/dinov3/HOW_TO_GET_DINOV3.md).
3. **The frozen run's results folder**, if the LoRA-against-frozen table is wanted. The copy in this repository's `results/` folder works.

Then, with the paths filled in:

```
python dinov3\train_dinov3_lora.py --train-val <Train_Val_Dataset> --qa <QA_download.csv> --out <results folder> --frozen-results ..\..\results\mangrove_segmentation\dinov3
```

If the code sits in the team's SDP folder layout, plain `python 2_Code\dinov3\train_dinov3_lora.py` finds everything by itself.

Useful variations:

| Add this | Effect |
| --- | --- |
| `--folds 1` | One fold only, as a first trial |
| `--lora-r 16` | Double the rank (strength follows automatically) |
| `--lora-targets q_proj k_proj v_proj o_proj` | Adapt all four attention layer types |
| `--lora-last-n 12` | Adapt only the last 12 blocks |
| `--no-grad-checkpoint --gpu-batch 4` | About 25% faster, 5.1 GB instead of 2.9 GB |
| `--scale 4` | Show DINOv3 a 4× enlarged chip. See the note below |
| `--model-id facebook/dinov3-vits16-pretrain-lvd1689m` | The small DINOv3 model |

Results appear in `<results folder>/lora_dinov3-vitl16-pretrain-sat493m/`. Open `RESULTS.md` first. A run with non-default LoRA settings gets its own folder name, for example `..._r16_a32_q-v`.

**Higher input resolution.** The Meeting 6 minutes say "fine-tune DINOv3 with LoRA at higher input resolution". `--scale 4` does that. It is not the default, because the frozen run used 2× and changing it would make the comparison unfair: a difference could come from the resolution, not from LoRA. To test resolution properly, the frozen run should also be repeated at `--scale 4`.

## 5. GPU memory and time

All numbers below were measured on this PC (RTX 5060 Ti, 8 GB) with a **random-weight model of the same size as DINOv3 ViT-L** (304 million numbers). Memory and speed do not depend on what the weights are, so these should carry over to the real model. Memory is what PyTorch used at its peak. Windows and other programs used another 0.2 to 1.9 GB of the 8 GB at the time.

| Setting | Peak GPU memory | Training time per chip |
| --- | --- | --- |
| **Default: 8 chips at once, gradient checkpointing on** | **2.9 GB** | **0.13 s** |
| 4 chips at once, checkpointing on | 2.1 GB | 0.12 s |
| 4 chips at once, checkpointing off | 5.1 GB | 0.09 s |
| 2 chips at once, checkpointing off | 3.4 GB | 0.09 s |
| 8 chips at once, checkpointing off | Does not fit (not finished after 200 s) | |
| 2 chips at once, checkpointing off, mixed precision off | 3.8 GB | 0.22 s |
| Rank 16 on all six layer types, 2 chips at once, checkpointing off | 4.6 GB | 0.11 s |
| Default at `--scale 4` (higher resolution) | 6.1 GB | 0.83 s |
| Small model (ViT-S), 8 chips at once, checkpointing off | 2.2 GB | 0.02 s |

**Whole-run time.** One complete fold was also run end to end on the GPU with the random ViT-L-size model and 215 made-up chips of the real size: 119 training chips and 38 validation chips took **18 seconds per epoch**, peak 2.9 GB.

The real folds have 56 to 107 training chips, so expect roughly 10 to 17 seconds per epoch. At the maximum of 40 epochs that is 7 to 11 minutes per fold, so **at most about an hour for all 5 folds, and less if early stopping triggers**. At `--scale 4` it would be about six times longer. These are estimates from the measurements above, not a timed real run.

Disk space: the DINOv3 ViT-L download is about 1.2 GB. Each fold saves a small file of a few megabytes (the decoder and the LoRA numbers only, not a copy of DINOv3).

## 6. Can it run on this PC?

**Yes for the hardware and software. Not yet for the real experiment, because the data and the weights are missing.**

| | Status |
| --- | --- |
| NVIDIA GPU | Yes: GeForce RTX 5060 Ti, 8 GB |
| PyTorch sees the GPU | Yes: PyTorch 2.11.0 with CUDA 12.8. Training steps ran on it |
| Enough GPU memory | Yes: 2.9 GB needed with the defaults |
| Python environment | Yes: conda environment `mangrove`, all packages import, 20 of 20 tests pass |
| Hugging Face downloads | Work (tested with a public file) |
| DINOv3 weights | **Not on this PC.** Licence not accepted and no login yet |
| Train/Val dataset and haze table | **Not on this PC** |

### How this PC was set up, and three things that had to be worked around

Miniconda was installed for this user only, in `C:\Users\Zain\miniconda3`, using winget. The `mangrove` environment was created from `environment.yml`, which was not edited. PyTorch (GPU build), `requirements_dinov3.txt` and `peft` were then installed into it with pip. Miniconda plus the environment take about 9 GB.

Three problems came up. Each fix is inside conda or the environment, not in the repository.

1. **Norton Antivirus inspects secure web traffic** with its own certificate, which conda and Python do not know, so downloads failed. Fix: conda was told to use the Windows certificate store (`ssl_verify: truststore` in `C:\Users\Zain\.condarc`), and the package `pip-system-certs` was added to the environment so Python does the same. Certificate checking is still on.
2. **Windows Smart App Control is on** and blocks some unsigned program files. It blocked files inside five conda-forge packages. Smart App Control was **not** switched off. The five packages were replaced with the official PyPI builds, which it accepts: pandas (3.0.5 in place of 3.1.1), and the same versions of matplotlib, scikit-learn, rasterio and pydantic-core. A newly installed package may be blocked the first time it loads and work on the second try.
3. **Two copies of a threading library clashed** (one from PyTorch, one from conda's maths library) and crashed the scoring step. Fix: the environment variable `MKL_THREADING_LAYER=TBB`, stored in the environment itself. It is applied by `conda activate mangrove`, which is one reason activating matters.

So the environment is the one in `environment.yml`, plus PyTorch and the DINOv3 packages, with five packages swapped for their PyPI builds. A teammate on a PC without Norton or Smart App Control would not need points 1 and 2.

VS Code also created a `.vscode/` folder in the repository when it noticed the new environment. It is not part of this work and should not be committed.

## 7. What was tested

**All 20 tests pass in the `mangrove` environment:** the 6 original tests, the 4 SVM tests, and 10 new LoRA tests. The LoRA tests use a tiny random DINOv3 (4 blocks) and made-up chips, and check that:

- The defaults are rank 8, strength 16, two layer types, all blocks, and only LoRA numbers are trainable.
- Rank, strength, layer types and number of blocks can all be changed, and a wrong layer name gives a clear error.
- Before training, the LoRA model's output equals the frozen model's.
- Training changes only the LoRA numbers and the decoder, never the original weights, with and without gradient checkpointing.
- The number of weight updates follows `--batch`, not `--gpu-batch`.
- A full LoRA run writes every file the frozen run writes, with the same columns, the same folds, the same training, validation and held-out chips, and the same scored pixels. The frozen script is run in the same test for this comparison.
- The results table has the same columns as the real `results/mangrove_segmentation/dinov3/cv_results.csv`.
- Non-default settings get their own results folder.
- The script refuses to run on a CPU unless forced, and the benchmark runs without data or weights.

**Also checked on the GPU, outside the test suite:** the benchmarks in section 5, and two end-to-end runs of one fold with a ViT-L-size random model (default settings, and checkpointing off with 4 chips at a time). Both finished and wrote all output files.

## 8. What is UNVERIFIED

1. **Whether LoRA improves the score.** No real run has been done. The scores in the tests come from a random model and mean nothing.
2. **Loading the real DINOv3 weights.** The gated download has not been tried on this PC. The layer names the script looks for (`q_proj`, `v_proj`) were confirmed on the model class in the installed `transformers` 5.23, not on the released checkpoint.
3. **Memory and time on the real model and data.** The numbers come from a random model of the published ViT-L size. The time for a whole run is an estimate.
4. **Whether the default settings are good ones.** They are sensible starting values from the LoRA paper and the existing code. None was tuned on our data.
5. **Mixed precision and gradient checkpointing on real data.** Both ran without errors on the GPU with random weights. Their effect on the final score has not been measured. They can be switched off with `--no-amp` and `--no-grad-checkpoint`.
6. **Other library versions.** Tested with `transformers` 5.23.0, `peft` 0.21.3 and PyTorch 2.11.0. The project asks for `transformers` 4.56 or later; older versions in that range were not tried.
7. **Whether the frozen results in the repository came from exactly this code.** If they came from a different version of `train_dinov3.py`, the LoRA-against-frozen table is less strict than it looks. Rerunning the frozen mode on the same PC would settle it; it takes minutes.
8. **The rest of the pipeline in this environment.** The download and chip-making steps were not run. rasterio and geopandas were only given a small read and write check.

## 9. What is left to do

1. Get the Train/Val dataset and the haze table from the team's shared storage.
2. Accept the DINOv3 licence and log in to Hugging Face.
3. Run the frozen mode on this PC, then the LoRA script, with `--folds 1` first.
4. Run all 5 folds and read the LoRA-against-frozen table. Differences under about 0.03 IoU are within noise, because each fold is one training run.
5. Decide with the supervisor whether to also test `--scale 4`, for both frozen and LoRA.
6. Review the three new files, then commit them.
