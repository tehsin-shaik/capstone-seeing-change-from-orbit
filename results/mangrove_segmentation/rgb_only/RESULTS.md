# RGB only into every model (Train/Val only)

In our committed runs the random forest and the SVM get all 8 channels per pixel (B02, B03, B04, B08, B11, B12, NDVI, MNDWI). DINOv3 only takes a 3 channel picture, so it gets B04, B03, B02. But our decoder also has a small convolution branch that reads all 8 channels of the chip, so the committed DINOv3 runs are "RGB into DINOv3, all 8 channels into the decoder branch". Here we give every model only B04, B03, B02 and change nothing else: same 5 leave-one-site-out folds, variant (b) (hazy scenes dropped from training), same settings and seeds, GMW v4.0.19 label, 2020 chips (215), scored with `evaluate.py`. Test_Dataset was not opened.

Mean IoU ± sd over the 5 held-out folds. The sd is the sample sd.

| Model | What the model gets in the RGB only run | RGB only | Committed 8 channel run | Difference |
|---|---|---|---|---|
| Random forest | B04, B03, B02 of each pixel | 0.559 ± 0.086 | 0.704 ± 0.101 | -0.145 |
| SVM, approximate RBF | B04, B03, B02 of each pixel | 0.527 ± 0.089 | 0.702 ± 0.105 | -0.175 |
| DINOv3 + LoRA, decoder branch removed | RGB into DINOv3. The decoder only gets the DINOv3 features, so nothing else reaches the model | 0.583 ± 0.116 | 0.694 ± 0.112 | -0.112 |
| DINOv3 + LoRA, decoder branch kept but fed RGB | RGB into DINOv3. The decoder branch is still there but reads B04, B03, B02 instead of all 8 | 0.626 ± 0.095 | 0.694 ± 0.112 | -0.068 |
| DINOv3 frozen, decoder branch removed | RGB into DINOv3 (no LoRA, backbone not trained). The decoder only gets the DINOv3 features | 0.553 ± 0.118 | 0.686 ± 0.107 | -0.133 |

For the three DINOv3 rows the committed run is the one with RGB into DINOv3 and all 8 channels into the decoder branch. The IoU of each fold is in `summary.csv`.

## What this shows

Every model gets worse with RGB only, by 0.07 to 0.18 mean IoU and on every fold, so the infrared bands and the two indices matter for all of them, and in the committed DINOv3 runs a good part of the score came through the 8 channel decoder branch. When every model gets the same RGB input, DINOv3 + LoRA is in front of the random forest: 0.583 against 0.559 with the branch removed (in front on 3 of the 5 folds, mostly on fold 1 with its 8 chips) and 0.626 against 0.559 with the branch kept on RGB (in front on all 5 folds). Frozen DINOv3 (0.553) is level with the random forest. With all 8 channels the order was the other way round (random forest 0.704, DINOv3 + LoRA 0.694), so on equal input DINOv3 + LoRA does a little better than the committed table suggests.

## What this does not show

There is one seed per fold and the folds are very different from each other, so a difference under about 0.03 between two models is within noise. That covers the 0.024 gap between DINOv3 + LoRA without the branch and the random forest. Keeping the branch on RGB gives 0.044 more than removing it, although it adds no new channels. UNVERIFIED: our guess is that the branch helps because it reads the reflectance at full 10 m resolution, while DINOv3 gets a contrast stretched picture cut into 16 pixel patches. We did not test this. The settings are the ones tuned for 8 channels. For the SVM this means the kernel width stays at 1/8 with 3 features, and the SVM picked the largest C on its grid on all 5 folds, so it could probably do a bit better on RGB with a wider grid. We left this alone because the task was to change only the input channels.

## Checks

Before the full runs we ran the new random forest and SVM script with all 8 channels and it gave the committed IoU on every fold, to 4 decimals. For DINOv3 + LoRA we retrained fold 1 with all 8 channels and the branch on. The committed number is 0.8417. We got 0.8424 with the script as it was before our edit and 0.8435 after the edit. So fold 1 does not come back exactly even with the untouched script, because training on the GPU does not give the same number twice. To check the edit itself we ran a short test on the CPU with the script before and after the edit, and there every output file was identical.

## How to get the numbers again

`src/mangrove_segmentation/rgb_only.py baselines` trains the random forest and the SVM on RGB, and `rgb_only.py summary` writes `summary.csv` from the run folders. The DINOv3 rows come from `dinov3/train_dinov3_lora.py --no-spectral`, `dinov3/train_dinov3_lora.py --rgb-branch` and `dinov3/train_dinov3.py --mode frozen --no-spectral`. Predictions, weights, logs and the per chip scores are kept outside the repo.
