"""Download COCO 2017 (val by default) + annotations into data/coco/.

Default = val2017 (~1GB, 5000 images) + annotations (~250MB). We carve BOTH the
train and val subsets out of val2017 (see data/subset.py), so you don't need the
18GB train2017 download to get the whole loop running. Pass --full to also fetch
train2017 for a larger experiment.

URLs (official COCO mirror):
    http://images.cocodataset.org/zips/val2017.zip
    http://images.cocodataset.org/zips/train2017.zip
    http://images.cocodataset.org/annotations/annotations_trainval2017.zip

Run: uv run python scripts/download_coco.py [--full] [--root data/coco] [--force]
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

COCO_URLS = {
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
    "train2017": "http://images.cocodataset.org/zips/train2017.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}


def download_file(url: str, dest: Path) -> None:
    """Stream a URL to ``dest`` with a progress bar (no whole-file buffering)."""
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, unit_divisor=1024, desc=f"download {dest.name}"
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MiB chunks
                f.write(chunk)
                bar.update(len(chunk))


def unzip(zip_path: Path, dest_dir: Path) -> None:
    """Extract ``zip_path`` into ``dest_dir`` (the COCO zips carry their own top folder)."""
    with zipfile.ZipFile(zip_path) as z:
        for member in tqdm(z.infolist(), desc=f"unzip {zip_path.name}", unit="file"):
            z.extract(member, dest_dir)


def fetch(name: str, root: Path, check_dir: Path, force: bool) -> None:
    """Download + unzip one component, skipping if it's already extracted."""
    if check_dir.exists() and not force:
        print(f"[skip] {name}: {check_dir} already present (use --force to redownload)")
        return
    zip_path = root / f"{name}.zip"
    download_file(COCO_URLS[name], zip_path)
    unzip(zip_path, root)
    zip_path.unlink(missing_ok=True)  # drop the zip once extracted
    print(f"[done] {name} -> {check_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download COCO 2017 (val + annotations by default)")
    parser.add_argument("--root", default="data/coco", type=Path, help="destination directory")
    parser.add_argument("--full", action="store_true", help="also download train2017 (~18GB)")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    root: Path = args.root
    root.mkdir(parents=True, exist_ok=True)

    # (component name, directory that proves it's already extracted)
    targets = [
        ("val2017", root / "val2017"),
        ("annotations", root / "annotations"),
    ]
    if args.full:
        targets.append(("train2017", root / "train2017"))

    for name, check_dir in targets:
        fetch(name, root, check_dir, args.force)

    ann_file = root / "annotations" / "instances_val2017.json"
    print("\nAll set. Point your config's data section at:")
    print(f"  images_dir: {root / 'val2017'}")
    print(f"  ann_file:   {ann_file}")
    if not ann_file.exists():
        print("  (warning: instances_val2017.json not found — check the download)")


if __name__ == "__main__":
    main()
