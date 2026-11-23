# Results

What each study found, and what it means for the project. Unless a page says otherwise, everything uses the
Train/Val data only (the locked test set has not been opened), 2020 chips scored against Global Mangrove Watch
(GMW v4.0.19), and each site is predicted by a model that never saw it (leave-one-site-out, 5 folds).

## Key takeaways

1. **The 2018 and 2025 images already line up** to about half a pixel, so registration is not what limits the change maps.
2. **Simple models are as good as big ones.** The random forest (about 0.70 IoU) ties the U-Net and DINOv3. The
   infrared bands matter for every model.
3. **GMW is a useful but imperfect label.** People agree with each other at 0.72–0.83 IoU, and GMW scores 0.70–0.75
   against each person. Against the people's majority label, the random forest (0.86) scores higher than GMW (0.74).
4. **One image date gives a noisy map, and the dates you pick change the answer.** Combining dates helps, and both
   years should use the same months.
5. **Change from 2018 to 2025:** with the same months in both years, Ghantoot (+8% to +17%) and Ras Ghurab (+6% to
   +21%) gain mangrove area. Eastern Mangroves and Jubail Island are unclear. This has not been checked against
   independent labels yet.

## Registration

| Study | What we found | What it means |
|---|---|---|
| [SIFT at Eastern Mangroves](registration/eastern_mangroves/RESULTS.md) | The Jubail registration check repeated at a second site gives a similar remaining error, about half a pixel. | The method works at more than one site. |
| [Phase correlation](registration/phase_correlation/RESULTS.md) | A method with no landmarks finds shifts of 0.8–4.5 m at the 6 sites. Al Nouf's result rests on one tile. | Two unrelated methods agree, so the small shifts are real and small. |
| [Al Aryam](registration/al_aryam/RESULTS.md) | The most shifted site (about 4.5 m). Correcting it removes about 3 ha of apparent gain, and the change is still inside same-year noise. | Even the worst shift doesn't change the conclusion. |
| [Different tides](registration/different_tides/RESULTS.md) | The same check on pairs with very different water levels, at Jubail and Eastern Mangroves. | Shows how much the tide affects the landmarks. Eastern is a weak test (small water difference). |

## Mangrove segmentation

| Study | What we found | What it means |
|---|---|---|
| [Baselines](mangrove_segmentation/baselines/RESULTS.md) | Random forest 0.70 IoU, NDVI rule 0.60. The forest removes most look-alike vegetation errors. | The bar every other model has to beat. |
| [SVM](mangrove_segmentation/svm/RESULTS.md) | The approximate RBF SVM ties the forest. The linear SVM is lower. | A non-linear model is needed, but it doesn't have to be deep. |
| [U-Net](mangrove_segmentation/unet/RESULTS.md) | 0.704, the same as the forest. | Deep learning gives no gain here yet. |
| [DINOv3](mangrove_segmentation/dinov3/RESULTS.md) and [with LoRA](mangrove_segmentation/dinov3_lora/RESULTS.md) | Frozen 0.686, LoRA 0.694. | A large pretrained model doesn't help on this little data. |
| [Model comparison](mangrove_segmentation/model_comparison/RESULTS.md) | RF, SVM, U-Net and both DINOv3 runs are tied: the biggest gap is 0.018 IoU, while sites differ by about 0.30. | The site matters far more than the model, so choose on cost and steadiness. |
| [Confidence](mangrove_segmentation/confidence/RESULTS.md) | Fold-to-fold spread is about 0.10. The chip bootstrap says nothing about new sites. | Differences under about 0.03 IoU are noise. |
| [Efficiency](mangrove_segmentation/efficiency/RESULTS.md) | A U-Net with a quarter of the parameters (1.9M) scores 0.699 vs 0.709, with half the GPU memory. | If a U-Net is used, the small one is enough. |
| [RGB only](mangrove_segmentation/rgb_only/RESULTS.md) | Every model drops 0.07–0.18 IoU without the infrared bands and indices. | Keep all 8 channels. |
| [Steadiness across dates](mangrove_segmentation/date_stability/RESULTS.md) | RF and SVM move least between dates. The deep models all have their worst date on the same image. | Another reason to prefer the forest for change maps. |
| [Multi-date vote](mangrove_segmentation/multi_date_vote/RESULTS.md) | Combining a chip's dates raises the forest from 0.714 to 0.752, mostly by removing one-off false alarms. | Combining dates is worth it. This led to the fusion window check. |
| [Model disagreement](mangrove_segmentation/model_disagreement/RESULTS.md) | Three different models all reject 9.0% of GMW's mangrove pixels. | Part of the models' "error" may be GMW's error. |
| [Hand labels](mangrove_segmentation/hand_labels/RESULTS.md) | People agree at 0.72–0.83; GMW 0.70–0.75 against each person; against the majority label RF 0.86, U-Net 0.82, GMW 0.74. | GMW is a little below human agreement, and the models are close to people. The majority label is our ground truth for these 6 chips. |

## Change detection

| Study | What we found | What it means |
|---|---|---|
| [First change maps](change_detection/RESULTS.md) | Mixing March and November, only Ghantoot (+25%) was above its same-year noise. | First pass. Replaced by the fusion window check. |
| [U-Net noise check](change_detection/unet/RESULTS.md) | The U-Net is not steadier across dates than the forest. | Swapping the model doesn't fix the noise. |
| [Date effect](change_detection/date_effect/RESULTS.md) | Tide explains about 1% of the same-year swing; differences between the images go with about half of it. Nothing tested makes the model steady. | The noise comes from the images, so the dates must be chosen and combined carefully. |
| [Fusion window](change_detection/fusion_window/RESULTS.md) | Same months a year apart agree almost as well as dates weeks apart. Averaging the probability over February–April is the most repeatable. Ghantoot +8% to +17%, Ras Ghurab +6% to +21%, the other two unclear. | Our candidate way of making change maps, pending independent validation. |
| [New bare ground](change_detection/human_activity/RESULTS.md) | At Jubail about 130 ha of new bare or built ground is above noise, and it is about twice as common near the mangroves. Ghantoot stays within noise. | There is human activity near Jubail's mangroves that could explain some of its change. |
