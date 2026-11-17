"""
Make a 6-band copy of a collect.py download so real_raster_check.py can read it.

collect.py (src/mangrove_segmentation) writes 7 bands: B02 B03 B04 B08 B11 B12 and SCL,
as float32 reflectance with NaN for no data, and band names like "B02 surface reflectance".
real_raster_check.py was written for Earth Engine exports: 6 bands named exactly
B2 B3 B4 B8 B11 B12, uint16 (reflectance x 10000) and no nodata value.

This script writes that layout:
- the SCL band is dropped
- reflectance is multiplied by 10000 and rounded to uint16
- negative values and NaN become 0 (dark water can be slightly negative after the offset)
- the grid, CRS and pixel positions are not changed

How to run:
  python src/registration/make_6band_copy.py <download.tif> <out_6band.tif>
"""

import os
import sys

import numpy as np
import rasterio

IN_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]   # names collect.py uses
OUT_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]      # names real_raster_check.py expects


def main(src_path, out_path):
    with rasterio.open(src_path) as src:
        # stop early if this isn't a collect.py file, otherwise the bands would be mislabelled
        for i, name in enumerate(IN_BANDS):
            if not (src.descriptions[i] or "").startswith(name):
                sys.exit(f"{src_path}: band {i + 1} is '{src.descriptions[i]}', expected {name}")
        refl = src.read(list(range(1, 7)))
        nan = np.isnan(refl)
        negative = refl < 0
        dn = np.rint(np.where(nan, 0, refl) * 10000.0)
        dn = np.clip(dn, 0, 65535).astype(np.uint16)
        profile = dict(driver="GTiff", dtype="uint16", count=6, width=src.width, height=src.height,
                       crs=src.crs, transform=src.transform, compress="deflate", tiled=True,
                       blockxsize=256, blockysize=256)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(dn)
            for i, name in enumerate(OUT_BANDS, start=1):
                dst.set_band_description(i, name)
            dst.update_tags(**src.tags())
            dst.update_tags(SIXBAND_COPY_OF=os.path.basename(src_path),
                            SIXBAND_NOTE="uint16 = reflectance x 10000; NaN and negatives set to 0; SCL dropped")
    print(f"wrote {out_path} ({dn.shape[2]} x {dn.shape[1]} px)")
    print(f"  no-data pixels set to 0: {nan[0].mean():.4%}")
    print("  negative values set to 0: " + ", ".join(f"{n} {negative[i].mean():.2%}" for i, n in enumerate(OUT_BANDS)))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
