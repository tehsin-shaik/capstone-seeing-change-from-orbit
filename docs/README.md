# Project Documentation

This directory contains the planning documents, technical documentation, research records, and reports produced throughout Seeing Change from Orbit.

- [Project plan](PROJECT_PLAN.md)
- [Progress log](PROGRESS.md)
- [Meeting minutes](meetings/README.md)
- [Project Proposal](proposal/SDP_Proposal_Seeing_Change_from_Orbit.pdf)
- [Data Dictionary](../data/DATA_DICTIONARY.md) — documents the satellite imagery, datasets, preprocessing, labelling, spatial reference systems, and data organization.
- [Hand Labels](../data/hand_labels/README.md) — manually annotated mangrove reference data.
- [Landmark Detection (Track A)](LANDMARK_DETECTION.md)
- [Registration Method (Track A)](REGISTRATION_METHOD.md)
- [`results/`](../results/) — Experiment results and evaluation reports are maintained in `results/`, with a `RESULTS.md` file for each experiment.

## Project Tracks

The project is organized around four connected tracks:

1. **Data preparation and synchronization**
   Find suitable labelled satellite datasets, preserve timestamps, align images spatially, and organize observations into comparable time periods.

2. **Landmark detection and regions of interest**
   Locate particular sites and define consistent geographic regions for comparison over time.

3. **Segmentation models**
   Investigate and compare models that identify mangroves and relevant boundaries.

4. **Combining methods and detecting change**
   Combine predictions and observations to distinguish persistent changes from noise, tides, and seasonal effects.

## Shared Goal

Reliably track and quantify mangrove changes through time using satellite imagery and computer vision.
