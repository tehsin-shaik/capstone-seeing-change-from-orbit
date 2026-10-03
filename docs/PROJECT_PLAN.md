# Project Plan

## SDP I Objective

Develop and apply the initial tools needed for satellite-image change detection using existing labelled datasets. If feasible, carry out a small Abu Dhabi case study using weak or self-supervised segmentation methods.

The work in SDP I should build the foundation for the core change and anomaly detection task in SDP II.

## Initial Milestones

| Milestone | Main activities | Owner(s) | Status |
| --- | --- | --- | --- |
| Targeted literature review | Review satellite imagery, segmentation, anomaly detection, and environmental monitoring methods. | Tehsin, Zayed, Zain, Ayesha | In progress |
| Dataset and resource investigation | Identify suitable labelled datasets, Abu Dhabi data sources, and available computing resources. | Zain, Zayed, Ayesha, Tehsin | In progress |
| Data preparation baseline | Export multispectral imagery, investigate timestamps, align images, and measure registration error. | Khaled, Tehsin, Zain | In progress |
| Segmentation investigation | Compare SAM, U-Net, and classical machine-learning approaches for mangrove segmentation. | Ayesha, Tehsin, Zain | In progress |
| Initial change-detection experiments | Compare aligned regions across time while accounting for tide, cloud, seasonal effects, and class imbalance. | Team | Exploratory analysis started |
| SDP I review | Summarize progress, limitations, results, and recommended next steps. | Team | Not started |

## Technical Tracks

1. Data preparation and synchronization
2. Segmentation models
3. Combining methods and detecting change
4. Landmark detection and regions of interest

## Current Technical Allocation

| Area | Owner(s) | Expected output |
| --- | --- | --- |
| Landmark selection and registration | Khaled, Tehsin | Aligned image pairs and measured registration error |
| SAM and DECDNet investigation | Ayesha, Tehsin | Foundation-model approach and DECDNet method summary |
| U-Net segmentation | Zayed | U-Net approach tested on selected imagery |
| Random forest and SVM baseline | Zain | Classical baseline using spectral bands, NDVI, and NDWI |
| Multispectral exports | Zain, Tehsin | Six-band Sentinel-2 exports for selected dates and sites |
| Ground-truth investigation | Zayed, Ayesha | Global Mangrove Watch review and manual-labelling plan |

## Working Practices

- Record work and decisions in GitHub.
- Use meaningful commits and link work to issues when appropriate.
- Keep meeting agendas and minutes.
- Update this plan when tasks, responsibilities, or dates change.