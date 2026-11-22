# Test Chips

Zayed added these chips with his [U-Net report](../../docs/reports/Seeing_Orbit_Mangrove_U-Net_Report.pdf) so that other segmentation models can be compared with his U-Net on the same held-out data. They are separate from the main pipeline: no script or result in this repository uses them.

`seeing-orbit-test-chips-inputs-only.zip` holds 10 Eastern Mangroves chips from November 2018 and November 2025, taken from an area the models were not trained on. Only the images are included. Zayed keeps the answer masks, so the chips can't be scored here.

To compare a model, run it on each chip and return one 256 × 256 mask per chip (1 = mangrove, 0 = non-mangrove), named after the chip, for example `patch_00010.npy`. Zayed scores the masks against his reference masks. Don't use these chips for training or tuning.

The chips have 6 channels (blue, green, red, near-infrared, NDVI, NDWI), not the 8 channels used by the main pipeline, so main-pipeline models need the chips converted first. The zip's `README.txt` gives the full format.
