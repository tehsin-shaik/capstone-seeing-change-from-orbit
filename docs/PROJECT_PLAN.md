# Project Plan

## SDP I Objective

Evaluate existing segmentation methods for mapping mangroves in Sentinel-2 imagery, including a foundation model (DINOv3), and use recent progress to improve one aspect of the task, in this project computational efficiency. The models are then used to produce 2018 to 2025 mangrove change maps for Abu Dhabi, which form the core of a mangrove change detection system. Segmentation is a step towards this system, not the final product.

### Deliverables for the final report

1. Aligned 2018 and 2025 imagery. The registration error is about half a pixel, checked with SIFT landmarks at Jubail Island, Eastern Mangroves and Al Aryam Islands, including pairs at different tides, and with phase correlation at all 6 training sites. Al Aryam Islands is the only site that needs a correction (about 5 m).
2. A comparison of six segmentation methods, all scored the same way on held-out sites: NDVI rule, random forest, SVM, U-Net, frozen DINOv3 and DINOv3 with LoRA.
3. An efficiency study. The top five models tie at about 0.70 IoU, and a U-Net with a quarter of the parameters and half the GPU memory matches the full U-Net (0.699 against 0.709 IoU).
4. 2018 to 2025 mangrove change maps, with a check for new bare or built ground around the mangroves at Jubail Island and Ghantoot to give context to the predicted change.
5. A critical analysis of where the models fail and why.

### Limitations

- **Labels.** The labels come from Global Mangrove Watch, which is machine-generated rather than annotated by experts, so no model can be claimed to be more accurate than GMW. On 6 hand-labelled chips, GMW agrees with each labeller (about 0.70 IoU) about as well as the two labellers agree with each other (0.687). If human-annotated labels become available, the models can be fine-tuned on them.
- **Consistency across dates.** Each model was run on two images from the same year, where change should be near zero. The result is not consistent: it depends on the date of the image. Tide explains only about 1% of these same-year swings, while brightness differences between images explain about half. Only Ghantoot gives a change estimate above this noise (about 25% more mangrove with the random forest and 37% with the U-Net), so it is the only change result reported with confidence.

## Milestones

| Milestone | Main activities | Owner(s) | Status |
| --- | --- | --- | --- |
| Targeted literature review | Review satellite imagery, segmentation, anomaly detection, and environmental monitoring methods. | Tehsin, Zayed, Zain, Ayesha | Done |
| Dataset and resource investigation | Identify suitable labelled datasets, Abu Dhabi data sources, and available computing resources. | Zain, Zayed, Ayesha, Tehsin | Done: Sentinel-2 L2A, GMW labels, 6 training sites, HPC GPU |
| Data preparation baseline | Export multispectral imagery, investigate timestamps, and build the training chips. | Ayesha, Zain, Tehsin | Done: [chip pipeline](../src/mangrove_segmentation/) and [data dictionary](../data/DATA_DICTIONARY.md) |
| Image registration | Detect landmarks, align images across dates, and measure registration error. | Khaled, Tehsin, Zain | Done: [registration method](REGISTRATION_METHOD.md), checked at all 6 sites with [phase correlation](../results/registration/phase_correlation/RESULTS.md), plus [Eastern Mangroves](../results/registration/eastern_mangroves/RESULTS.md), [different tides](../results/registration/different_tides/RESULTS.md) and [Al Aryam](../results/registration/al_aryam/RESULTS.md) |
| Segmentation investigation | Compare classical, U-Net and foundation-model approaches for mangrove segmentation. | Ayesha, Zain, Tehsin, Zayed | Done: 6 models compared; top 5 tied at about 0.70 IoU |
| Efficiency study | Reduce the size and GPU memory of the segmentation model without losing accuracy. | Tehsin | Done: [U-Net with 25% of the parameters and half the GPU memory matches the full one (0.699 vs 0.709 IoU)](../results/mangrove_segmentation/efficiency/RESULTS.md) |
| Change detection | Produce 2018 to 2025 change maps, measure the same-year swing, test what causes it, and check for new bare or built ground near the mangroves. | Zain, Tehsin | Done: [RF and U-Net change maps at all 6 sites](../results/change_detection/RESULTS.md); only Ghantoot is above the same-year swing ([U-Net](../results/change_detection/unet/RESULTS.md), [date effect](../results/change_detection/date_effect/RESULTS.md), [bare or built ground](../results/change_detection/human_activity/RESULTS.md)) |
| Failure analysis | Find where and why the models fail: across dates, between models and against GMW. | Tehsin, Zain | In progress: [date stability](../results/mangrove_segmentation/date_stability/RESULTS.md), [model disagreement](../results/mangrove_segmentation/model_disagreement/RESULTS.md), [multi-date vote](../results/mangrove_segmentation/multi_date_vote/RESULTS.md) and [date effect](../results/change_detection/date_effect/RESULTS.md); to be written up in the final report |
| Human ground truth | Hand-label 6 Train_Val chips and score GMW and the models against them. | Team | In progress: [6 chips labelled by Tehsin and Khaled](../data/hand_labels/README.md); [two people agree at 0.687 IoU, GMW 0.70 against each](../results/mangrove_segmentation/hand_labels/RESULTS.md) |
| SDP I final report and demo | Summarize the results, limitations and recommended next steps, and present a live demo. | Team | In progress |

## Technical Tracks

1. Data preparation and synchronization
2. Landmark detection and regions of interest
3. Segmentation models
4. Combining methods and detecting change

## Work Done So Far

| Area | Owner(s) | Output |
| --- | --- | --- |
| Landmark detection | Khaled | [Detector comparison](LANDMARK_DETECTION.md) |
| Image alignment and registration error | Tehsin | [Registration method](REGISTRATION_METHOD.md), [phase correlation cross-check](../results/registration/phase_correlation/RESULTS.md), [Al Aryam](../results/registration/al_aryam/RESULTS.md) |
| Registration at other sites and tides | Zain | [Eastern Mangroves](../results/registration/eastern_mangroves/RESULTS.md), [different tides](../results/registration/different_tides/RESULTS.md) |
| Data pipeline, NDVI rule, random forest, U-Net, frozen DINOv3 | Ayesha | [Pipeline](../src/mangrove_segmentation/), [baselines](../results/mangrove_segmentation/baselines/RESULTS.md) |
| U-Net report and test-chip inputs | Zayed | [U-Net report](reports/), [test chips](../data/test_chips/) |
| SVM, DINOv3 LoRA, data dictionary | Zain | [SVM](../results/mangrove_segmentation/svm/RESULTS.md), [LoRA](../results/mangrove_segmentation/dinov3_lora/RESULTS.md), [data dictionary](../data/DATA_DICTIONARY.md) |
| Change maps and noise checks | Zain | [Change detection](../results/change_detection/RESULTS.md) and its subfolders |
| Model comparison, U-Net efficiency, date stability | Tehsin | [Model comparison](../results/mangrove_segmentation/model_comparison/RESULTS.md), [efficiency](../results/mangrove_segmentation/efficiency/RESULTS.md), [date stability](../results/mangrove_segmentation/date_stability/RESULTS.md) |
| Hand labels and scoring against them | Tehsin (labels and scoring), Khaled (labels) | [data/hand_labels](../data/hand_labels/README.md), [hand-label scores](../results/mangrove_segmentation/hand_labels/RESULTS.md) |

## Remaining Work

| Task | Expected output |
| --- | --- |
| Finish the team's hand labels and merge them (agreed area or union) | One ground truth per chip |
| Pick the final model (RF or approx. RBF SVM) and run the Test_Dataset once | Final test scores |
| Try multi-month composites to reduce the same-year swing | Change maps that hold up at more sites than Ghantoot |
| SDP I final report, presentation and demo | Report, slides and live demo |

## Working Practices

- Record work and decisions in GitHub.
- Use meaningful commits and link work to issues when appropriate.
- Keep meeting agendas and minutes.
- Update this plan when tasks, responsibilities, or dates change.