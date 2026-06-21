"""Download COCO 2017 (val by default) + annotations into data/coco/.

Default = val2017 (~1GB, 5000 images) + annotations (~250MB). We carve BOTH the
train and val subsets out of val2017 (see data/subset.py), so you don't need the
18GB train2017 download to get the whole loop running. Pass --full to also fetch
train2017 for a larger experiment.

URLs (official COCO mirror):
    http://images.cocodataset.org/zips/val2017.zip
    http://images.cocodataset.org/zips/train2017.zip
    http://images.cocodataset.org/annotations/annotations_trainval2017.zip

Run: uv run python scripts/download_coco.py [--full] [--root data/coco]
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Download COCO 2017")
    parser.add_argument("--root", default="data/coco", help="destination directory")
    parser.add_argument("--full", action="store_true", help="also download train2017 (18GB)")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    # TODO:
    #   1. mkdir -p {root}/{annotations}.
    #   2. For each needed zip: stream-download with requests (show tqdm by content-length),
    #      skip if the target dir already exists and not --force.
    #   3. Unzip into root (val2017/, train2017/, annotations/).
    #   4. Print where things landed + the ann_file path to put in your config.
    raise NotImplementedError("Implement scripts/download_coco.py")


if __name__ == "__main__":
    main()
