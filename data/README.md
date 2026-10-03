# Project Data

This folder documents the datasets, satellite imagery, and reference data used throughout the project. Large raster files are stored in shared team storage and are not committed to the Git repository.

## Study Area

The initial study area is Eastern Mangroves, Abu Dhabi. Additional locations, such as Jubail and Reem Island, may be included in later experiments.

## Satellite Imagery

The project primarily uses Sentinel-2 multispectral imagery to study changes in mangrove extent over time. 

Different image pairs may be used for exploratory analysis, testing, and the main long-term comparison. These experiments should be documented separately, including their acquisition dates, processing details, and intended purpose.

## Sentinel-2 Bands

The current multispectral plan uses:

| Band | Measurement | Native resolution |
| --- | --- | --- |
| B2 | Blue | 10 m |
| B3 | Green | 10 m |
| B4 | Red | 10 m |
| B8 | Near-infrared | 10 m |
| B11 | Shortwave infrared 1 | 20 m |
| B12 | Shortwave infrared 2 | 20 m |

B11 and B12 must be resampled when they are placed on the common 10 m grid; this changes the grid spacing but does not create additional spatial detail.

The exact Sentinel collection or product ID, processing level, scale factors, no-data value, cloud-filtering criteria, and export script still need to be recorded before the dataset is considered reproducible.

## External Raster Data

Large satellite rasters and other source datasets are stored outside Git.

For each external dataset, the project should maintain metadata such as:

- File or dataset name
- Acquisition date or time period
- Study area
- Source and product/collection
- Processing level
- Spatial resolution
- Coordinate reference system
- File size
- SHA-256 checksum
- Download or export instructions
- Licence and attribution requirements

This information allows the data used in experiments to be identified and reproduced without storing large raster files in the repository.

## Reference and Benchmark Data

- Sentinel-1 radar imagery
- Sentinel-2 optical imagery
- Global Mangrove Watch reference maps
- LEVIR-CD and S2Looking change-detection benchmarks
- Other labelled datasets identified through the literature review

Datasets will be selected based on their relevance to mangrove mapping, change detection, segmentation, and model evaluation.

## Data Organization

Data should be separated according to their role in the project:

- Source data: original satellite imagery and reference datasets
- Processed data: resampled, aligned, or otherwise preprocessed imagery
- Training data: regions used for model development
- Validation data: regions used for model selection and tuning
- Test data: held-out regions used for final evaluation

Training, validation, and test regions should remain geographically separate to reduce spatial leakage and provide a more meaningful evaluation of model generalization.

A final hand-verified test set should be frozen before final evaluation and must not be used for training or model development.

## Repository Data Policy

- Do not commit raw or processed satellite rasters, dataset archives, model weights, or credentials to normal Git history.
- Commit source metadata, licences, checksums, download or export instructions, and reproducible preprocessing code.
- Small sample patches may be added later only when their licence permits redistribution and their purpose is documented.
- Training, validation, and test regions must remain geographically separate.
- A future hand-verified test set must be frozen before final evaluation and must never be used for training.
