# Project Plan

## SDP I Objective

Develop and apply the initial tools needed for satellite-image change detection using existing labelled datasets. If feasible, carry out a small Abu Dhabi case study using weak or self-supervised segmentation methods.

The work in SDP I should build the foundation for the core change and anomaly detection task in SDP II.

## Milestones

| Milestone | Main activities | Owner(s) | Status |
| --- | --- | --- | --- |
| Targeted literature review | Review satellite imagery, segmentation, anomaly detection, and environmental monitoring methods. | Tehsin, Zayed, Zain, Ayesha | Done: [proposal](proposal/) |
| Dataset and resource investigation | Identify suitable labelled datasets, Abu Dhabi data sources, and available computing resources. | Zain, Zayed, Ayesha, Tehsin | Done: Sentinel-2 L2A, GMW labels, 6 training sites, HPC GPU |
| Data preparation baseline | Export multispectral imagery, investigate timestamps, and build the training chips. | Ayesha, Zain, Tehsin | Done: [chip pipeline](../src/mangrove_segmentation/) and [data dictionary](../data/DATA_DICTIONARY.md) |
| Image registration | Detect landmarks, align images across dates, and measure registration error. | Khaled, Tehsin, Zain | Done: [registration method](REGISTRATION_METHOD.md), checked at all 6 sites with [phase correlation](../results/registration/phase_correlation/RESULTS.md), plus [Eastern Mangroves](../results/registration/eastern_mangroves/RESULTS.md), [different tides](../results/registration/different_tides/RESULTS.md) and [Al Aryam](../results/registration/al_aryam/RESULTS.md) |
| Segmentation investigation | Compare classical, U-Net and foundation-model approaches for mangrove segmentation. | Ayesha, Zain, Tehsin, Zayed | Done: 6 models compared; top 5 tied at about 0.70 IoU |
| Initial change-detection experiments | Compare aligned regions across time while accounting for tide, cloud, seasonal effects, and class imbalance. | Zain, Tehsin | In progress: first maps; at 5 of 6 sites change is within the same-year swing |
| Human ground truth | Hand-label test chips and score GMW and the models against them. | Tehsin | Done: [6 chips labelled by Tehsin and Khaled](../data/hand_labels/README.md); [two people agree at 0.687 IoU, GMW 0.70 against each](../results/mangrove_segmentation/hand_labels/RESULTS.md) |
| SDP I review | Summarize progress, limitations, results, and recommended next steps. | Team | In progress |

## Technical Tracks

1. Data preparation and synchronization
2. Segmentation models
3. Combining methods and detecting change
4. Landmark detection and regions of interest

## Work Done So Far

| Area | Owner(s) | Output |
| --- | --- | --- |
| Landmark detection | Khaled | [Detector comparison](LANDMARK_DETECTION.md) |
| Image alignment and registration error | Tehsin | [Registration method](REGISTRATION_METHOD.md), [phase correlation cross-check](../results/registration/phase_correlation/RESULTS.md), [Al Aryam](../results/registration/al_aryam/RESULTS.md) |
| Registration at other sites and tides | Zain | [Eastern Mangroves](../results/registration/eastern_mangroves/RESULTS.md), [different tides](../results/registration/different_tides/RESULTS.md) |
| Data pipeline, NDVI rule, random forest, U-Net, frozen DINOv3 | Ayesha | [Pipeline](../src/mangrove_segmentation/), [baselines](../results/mangrove_segmentation/baselines/RESULTS.md) |
| U-Net report and test-chip inputs | Zayed | [U-Net report](reports/), [test chips](../data/test_chips/) |
| SVM, DINOv3 LoRA, data dictionary, proposal | Zain | [SVM](../results/mangrove_segmentation/svm/RESULTS.md), [LoRA](../results/mangrove_segmentation/dinov3_lora/RESULTS.md), [data dictionary](../data/DATA_DICTIONARY.md) |
| Change maps and noise checks | Zain | [Change detection](../results/change_detection/RESULTS.md) and its subfolders |
| Model comparison, U-Net efficiency, date stability | Tehsin | [Model comparison](../results/mangrove_segmentation/model_comparison/RESULTS.md), [efficiency](../results/mangrove_segmentation/efficiency/RESULTS.md), [date stability](../results/mangrove_segmentation/date_stability/RESULTS.md) |
| Hand labels and scoring against them | Tehsin | [data/hand_labels](../data/hand_labels/README.md), [hand-label scores](../results/mangrove_segmentation/hand_labels/RESULTS.md) |

## Remaining Work

| Task | Expected output |
| --- | --- |
| Merge the two people's labels (agreed area or union) | One ground truth per chip |
| Pick the final model (RF or approx. RBF SVM) and run the Test_Dataset once | Final test scores |
| Change maps that hold up against the same-year swing | Maps from multi-date votes or multi-month composites |
| SDP I report and demo | Report and live demo |

## Working Practices

- Record work and decisions in GitHub.
- Use meaningful commits and link work to issues when appropriate.
- Keep meeting agendas and minutes.
- Update this plan when tasks, responsibilities, or dates change.