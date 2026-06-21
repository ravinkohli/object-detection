"""Precompute Selective Search proposals for Fast R-CNN and cache them to disk.

Selective Search is slow (~1-2s/image), so we run it once over the subset images and
save the results; training then just loads the cache. Only needed for Fast R-CNN
(Faster R-CNN learns its proposals via the RPN).

Run: uv run python scripts/precompute_proposals.py --config configs/fast_rcnn.yaml
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute Selective Search proposals")
    parser.add_argument("--config", required=True, help="path to a fast_rcnn config")
    args = parser.parse_args()

    # TODO:
    #   1. cfg = utils.config.load_config(args.config).
    #   2. Resolve the subset image ids (data/subset.select_image_ids) so you only
    #      compute proposals for images you'll actually train/eval on.
    #   3. data.proposals.cache_proposals(ann_file, images_dir, cache_dir, image_ids,
    #                                     fast=cfg.proposals.fast, max_proposals=...).
    raise NotImplementedError("Implement scripts/precompute_proposals.py")


if __name__ == "__main__":
    main()
