# Seeing Change from Orbit

**Detecting Coastal and Urban Change over Abu Dhabi from Satellite Imagery**

Senior Design Project - Khalifa University

Supervisor: Dr. Panos Liatsis

## Overview

This project develops a satellite-image change detection system for
Abu Dhabi's coastline and urban areas, with a focus on distinguishing
meaningful changes such as urban development and mangrove change from
variations caused by tides, seasons, clouds, and illumination.

## Documentation

- [Project plan](docs/PROJECT_PLAN.md)
- [Progress log](docs/PROGRESS.md)
- [Meeting minutes](docs/meetings/README.md)
- [Proposal](docs/proposal/SDP_Proposal_Seeing_Change_from_Orbit.pdf)
- [Data documentation](data/README.md) and [data dictionary](docs/DATA_DICTIONARY.md)
- [Registration method (Track A)](docs/REGISTRATION_METHOD.md)
- [U-Net report](docs/reports/Seeing_Orbit_Mangrove_U-Net_Report.pdf)

## Repository layout

| Folder | What's in it |
| --- | --- |
| `src/registration/` | Track A: landmarks and image alignment (tests in `src/registration/tests/`) |
| `src/mangrove_segmentation/` | Track B: data prep, all segmentation models, scoring (tests in `tests/` inside it) |
| `src/change_detection/` | 2018 to 2025 change maps and the bare/built ground check |
| `results/` | One folder per track, each experiment with its own `RESULTS.md` |
| `data/` | Small data only: hand labels, test chips. Big rasters are kept outside Git |
| `docs/` | Plan, progress, meetings, proposal, method write-ups and reports |
