# Progress Log

What the team did each month, with links to the code and results.

## August 2026

- Formed the SDP group.

## September 2026

### 2 to 14 September: Scope and Plan

- Agreed the scope: mangrove change detection over Abu Dhabi, with Sentinel-1 and Sentinel-2 as the main data sources ([Meeting 1](meetings/2026-09-02_Meeting_1_Project_Scope_And_Direction.pdf)).
- Set up the team GitHub repository and README, so each member's work is recorded under their own name (Tehsin, 9 September).
- Set the SDP I aim and split a targeted literature review between segmentation and change detection ([Meeting 2](meetings/2026-09-14_Meeting_2_First_SDP1_Supervisor_Meeting.pdf)).
- Decided the core workflow: line up images by date, pick a coastal region around the mangroves anchored on fixed landmarks, segment the mangroves inside it and compare them over time, using several images per period rather than one because of cloud and tide ([Meeting 2](meetings/2026-09-14_Meeting_2_First_SDP1_Supervisor_Meeting.pdf)).

### 19 to 28 September: Data, Tools and Task Allocation

- Discussed the plan and broke it down into four project tracks: data preparation and synchronization, landmark detection and regions of interest, segmentation models, and combining methods and detecting change.
- Agreed that the imagery must be multispectral (B8, B11 and B12 added), that the first deliverable is a data dictionary (a reference document describing each data source: its files, bands, units, resolution, coordinate system and dates), and that 2018 against 2025 with GMW labels is the main comparison ([Meeting 3](meetings/2026-09-21_Meeting_3_Data_Preparation_And_Fusion_Planning.pdf)).
- Set up the team's GitHub workflow and walked everyone through cloning in VS Code, so contributions are visible every week (Tehsin, [27 September](meetings/2026-09-27_Team_Meeting_Preparation_For_Meeting_4.pdf)).
- Exported the first 6-band Sentinel-2 images (red, green, blue, near-infrared and two shortwave-infrared bands) of Jubail for 2018 and 2025, did the first QGIS work on them, and wrote the first data dictionary (Zain).
- Reviewed the segmentation literature and pointed out that registration had to be studied before segmentation (Ayesha).
- Found DECDNet (Maung et al., 2026), the closest published work, which became the reference design: red, green, blue, near-infrared, NDVI and NDWI inputs, one encoder per date, and U-Net baselines (Tehsin).
- Drafted the semester Gantt chart, with weekly tasks from data collection to validation (Khaled).
- Agreed to register the images before segmenting, and to compare three segmentation routes: conventional machine learning, U-Net and a foundation model. Tasks were allocated: landmark detection (Khaled), alignment and registration error (Tehsin), SAM and DECDNet (Ayesha), U-Net (Zayed), RF and SVM (Zain) ([Meeting 4](meetings/2026-09-28_Meeting_4_Registration_Segmentation_And_Task_Allocation.pdf)).

### 30 September: Eastern Mangroves Exploratory Comparison

- Compared Eastern Mangroves imagery from 2 November 2023 and 1 November 2025 using NDVI and NDWI difference maps. The candidate-change mask covered 15.5% of valid pixels, but it also picks up tide, season, lighting and small misalignments, so it was only a first look. Proper registration followed in October (see [Registration method](REGISTRATION_METHOD.md)).

![Eastern Mangroves exploratory comparison](images/eastern-mangroves-visual-inspection-2023-2025.jpg)

## October 2026

### 1 to 3 October: Six-Band Exports, SAM and Registration Review

- Exported 6-band Sentinel-2 images of Eastern Mangroves for 2018 and 2025, documented with checksums so the files can be checked and reused (Tehsin).
- Tested SAM ViT-H for mangrove segmentation, with its decoder fine-tuned on GMW labels and NDVI and NDWI as prompts (Ayesha). The prompts raised IoU from 0.09 to 0.36, but a plain NDVI threshold reached 0.41, so the indices did most of the work and the team moved on to a U-Net.
- Reviewed the literature on landmark detection, feature matching and co-registration for Sentinel-1 (radar) and Sentinel-2 (optical) imagery (Tehsin and Khaled).
- Set QGIS as the tool for quality checks, registration checks, hand labelling and maps, not for model training.

### 4 October: Track A Landmarks and Alignment, and the U-Net Report

- Compared 6 corner detectors (Harris, Shi-Tomasi, FAST, ORB, SIFT and AKAZE) and 3 search areas for matching the 2018 and 2025 Jubail images (Khaled). SIFT with open water ignored was the most reliable, with a worst error of 0.18 px. The textbook "built-up land only" rule did worst, because built-up land is what changed. See [LANDMARK_DETECTION.md](LANDMARK_DETECTION.md).
- Built the image alignment and registration-error method (Tehsin): SIFT landmarks with RANSAC, similarity and affine fits, and a correction is applied only if it lowers the held-out error by at least 1 m. At Jubail the error with no correction is about 5 m, half a pixel, so no correction was needed. Over 2,000 trials, 3 landmarks with an affine fit undo a known misalignment of about 190 m to about 9 m, against 28 m with 2 landmarks. The method was also checked on the real 6-band rasters. See [REGISTRATION_METHOD.md](REGISTRATION_METHOD.md).
- Built a six-channel U-Net workflow for Eastern Mangroves, 2018 to 2025, with QGIS for review and mapping, and wrote it up as a technical report (30 September); its preliminary geographic test reached an F1 of 0.814. Prepared the test chips that go with it (Zayed). See [reports](reports/).

### 5 October: SAM Results Review

- Reviewed the SAM results and agreed to add a U-Net baseline, consider multispectral-pretrained models and move to 10 m ground truth. Ayesha started the university HPC access request ([Meeting 5](meetings/2026-10-05_Meeting_5_SAM_Segmentation_Results.pdf)).

### 18 to 23 October: Segmentation Pipeline and First Models

- Built the mangrove segmentation pipeline (Ayesha): it downloads Sentinel-2 surface reflectance for each site and date onto one 10 m grid, applies the Sentinel-2 offset once, cuts 256 x 256 chips with 8 channels (six bands, NDVI and MNDWI), attaches the GMW labels and records every chip in a catalog. See [src/mangrove_segmentation](../src/mangrove_segmentation/).
- Set up leave-one-site-out cross-validation on the 2020 chips (215 chips, 6 Abu Dhabi sites, 5 folds), so every score comes from a site the model never saw. The Test_Dataset stays closed until the final model is chosen.
- First results (23 October): NDVI + MNDWI rule, random forest, U-Net and frozen DINOv3. The random forest set the bar at 0.70 mean IoU. See [baselines](../results/mangrove_segmentation/baselines/RESULTS.md), [U-Net](../results/mangrove_segmentation/unet/RESULTS.md) and [DINOv3](../results/mangrove_segmentation/dinov3/RESULTS.md).

### 26 to 31 October: Model Comparison

- Presented the model comparison across the six sites (Ayesha): U-Net 0.71 IoU, random forest 0.70, frozen DINOv3 0.67, NDVI + MNDWI rule 0.60. Two GMW releases agree with each other at only 0.47 to 0.83 IoU, so about 0.70 is near what the labels allow ([Meeting 6](meetings/2026-10-26_Meeting_6_Mangrove_Model_Comparison.pdf)).
- Extended the registration work to more UAE mangrove sites (Tehsin).
- Reviewed how GMW v4.1 is made: an XGBoost classifier on Landsat and Sentinel-2 time series, so GMW is machine-generated, not hand-labelled (Team).
- Agreed to pick the final model from cross-validation only, taking the simpler model on a tie; to fine-tune DINOv3 with LoRA first; and to hand-label a small sample in pairs to check GMW.

## November 2026

### 2 to 16 November: Data, DINOv3 LoRA and First Hand Label

- Updated the [data dictionary](../data/DATA_DICTIONARY.md), describing the sites, imagery, chips, labels and file formats (Zain).
- Fine-tuned DINOv3 with LoRA for 80 epochs (Zain). It reached 0.694 IoU, 0.008 above frozen DINOv3 and not above the random forest (0.704). See [dinov3_lora](../results/mangrove_segmentation/dinov3_lora/RESULTS.md).
- Hand-labelled a Khor Al Beida chip, `KAB_2020_07` (Tehsin). See [data/hand_labels/Tehsin](../data/hand_labels/).
- Progress check: about three weeks remain in the semester, and the SDP I report is due at the end ([Meeting 7](meetings/2026-11-09_Meeting_7_Progress_Check_And_SDP1_Report.pdf)).

### 16 to 18 November: More Models, Registration Checks and First Change Maps

- Reviewed the alignment, the hand labels and the comparison of manual and model labels. Agreed to document the lack of expert labels as a limitation, and to write a status report with an alternative plan ([Meeting 8](meetings/2026-11-16_Meeting_8_Project_Status_And_Annotation_Limitations.pdf)).
- Restated the SDP I objective and agreed the five final-report deliverables and the limitations to report ([16 November team meeting](meetings/2026-11-16_Team_Meeting_SDP1_Objective_And_Final_Report_Deliverables.pdf)).

#### More Models

- Trained linear and approximate RBF SVM baselines (Zain). The RBF SVM reached 0.702 IoU, level with the random forest, while the linear SVM stayed at 0.60, no better than the NDVI rule. See [svm](../results/mangrove_segmentation/svm/RESULTS.md).
- Mapped where the random forest, SVM and LoRA all disagree with GMW (Zain). 72% of the GMW mangrove that all three reject lies on a GMW edge, and leaving those pixels out raises IoU from about 0.70 to 0.83, so much of the remaining error is at label edges. See [model_disagreement](../results/mangrove_segmentation/model_disagreement/RESULTS.md).
- Combined each chip's 2020 dates by majority vote (Zain). The random forest rises from 0.714 to 0.752 IoU, mostly because one-off false alarms are voted out. See [multi_date_vote](../results/mangrove_segmentation/multi_date_vote/RESULTS.md).

#### Registration Checks

- Re-ran the alignment method, unchanged, at Eastern Mangroves and on pairs with very different water levels, such as Jubail at 33% against 63% water (Zain). The error stayed at 5 to 6 m and no correction was needed, so the registration holds across sites and tides. See [eastern_mangroves](../results/registration/eastern_mangroves/RESULTS.md) and [different_tides](../results/registration/different_tides/RESULTS.md).

#### First Change Maps

- Produced 2018 to 2025 mangrove change maps for all 6 sites with the random forest, and measured the swing between two dates of the same year as the noise level (Zain). Only Ghantoot (+25%) changes more than that swing. See [change_detection](../results/change_detection/RESULTS.md).
- Repeated the change maps with the U-Net (Zain). It is not steadier across dates than the random forest, and Ghantoot is again the only site above the swing (+37%). See [unet](../results/change_detection/unet/RESULTS.md).
- Mapped new bare or built ground around the mangroves at Jubail Island and Ghantoot (Zain). At Jubail it is above the same-year noise in every zone, and twice as common within 500 m of the mangroves as further away. The rule cannot tell construction from sand. See [human_activity](../results/change_detection/human_activity/RESULTS.md).

### 19 to 22 November: Model Choice, Same-Year Swings, Registration Cross-Check and Hand Labels

- Set the SDP I priorities: a working pipeline, an RGB-to-RGB comparison of the random forest, SVM and foundation model, and consolidating the results for the report. Eight-band and radar inputs move to SDP II, the tide finding is to be presented as preliminary, and hand-label disagreements are to be resolved by a third labeller ([Meeting 9](meetings/2026-11-20_Meeting_9_Model_Results_Labelling_Protocol_And_SDP1_Priorities.pdf)).

#### Model Choice

- Paired comparison of all models across the 5 folds (Tehsin). RF, SVM (approx. RBF), U-Net, frozen DINOv3 and DINOv3 LoRA are tied: the largest gap is 0.018 IoU, against about 0.30 between the easiest and hardest site. Under the rule of taking the simpler model on a tie, the pick is RF or SVM. See [model_comparison](../results/mangrove_segmentation/model_comparison/RESULTS.md).
- U-Net efficiency study (Tehsin). Base 16 keeps 0.699 IoU, against 0.709 for the full model, with a quarter of the parameters and half the GPU memory. Base 4 drops to 0.616. See [efficiency](../results/mangrove_segmentation/efficiency/RESULTS.md).
- Measured how much each model's score moves between image dates (Tehsin). The random forest and SVM never drop below about 0.47 on any date, while the U-Net and DINOv3 drop to 0.33 to 0.37. Resampling the chips shows the 0.70 is accurate to about ±0.035. See [date_stability](../results/mangrove_segmentation/date_stability/RESULTS.md).

#### Why the Same-Year Swings Happen

- Tested what causes the same-year swings (Zain). Tide explains about 1%, brightness differences between the images about half, and no correction tested made the models steady. See [date_effect](../results/change_detection/date_effect/RESULTS.md).

#### Registration Cross-Check

- Phase correlation, a method that does not use landmarks, measured the shift at all six sites (Tehsin). The shifts are small. The largest is the Al Aryam March pair, at about 4.5 m and mostly north. See [phase_correlation](../results/registration/phase_correlation/RESULTS.md).
- Al Aryam was then registered and its change map redone (Tehsin). The shift removes about 3 ha of fake gain, but the net change is still within the same-year swing. See [al_aryam](../results/registration/al_aryam/RESULTS.md).

#### Hand Labels

- Hand-labelled 6 chips in QGIS, with uncertain areas marked separately: Jubail Island (3 chips), Ras Ghurab, Eastern Mangroves and Ghantoot (Tehsin, Khaled). See [data/hand_labels](../data/hand_labels/README.md).
- Compared Tehsin's labels with Khaled's on the same 6 chips, and scored GMW and the models against both (Tehsin). The two people agree at 0.687 IoU, and GMW scores 0.699 and 0.706 against them, so GMW is about as good as a second labeller. Khaled draws wider edges; the random forest is closest to Tehsin (0.845) and the U-Net to Khaled (0.742). See [hand_labels](../results/mangrove_segmentation/hand_labels/RESULTS.md).

## Next Steps

1. Finish the team's hand labels and merge them into one ground truth per chip: the agreed area or the union.
2. Choose between the random forest and the approximate RBF SVM, then run the Test_Dataset once.
3. Try multi-month composites to reduce the same-year swing.
4. Write the SDP I final report and prepare the presentation and demo.
