# Checksums

The Sentinel-2 exports are kept outside Git, so `make_checksums.py` records a fingerprint of each file instead. It walks a folder, including sub-folders, and writes a CSV with one row per file: `filename`, `size_bytes` and `sha256`. If even one byte of a file changes, its SHA-256 changes, so committing the CSV lets anyone check that they have exactly the files an experiment used. The script only reads files and needs no extra packages.

From `src/mangrove_segmentation`:

```
python make_checksums.py <exports folder> --out ../../data/checksums_exports.csv --pattern "*.tif"
```

Leave out `--pattern` to include every file. It hasn't been run on the real exports yet, so there is no checksum table in the repository.
