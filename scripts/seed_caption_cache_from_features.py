"""Populate the caption cache from an existing image-only feature cache."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.captioning import load_cache, save_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', default='results/features/stage_b_cross_source_rich.json')
    parser.add_argument('--cache', default='results/captions/detector_captions.json')
    args = parser.parse_args()
    cache_path = Path(args.cache)
    cache = load_cache(cache_path)
    added = 0
    for row in json.loads(Path(args.features).read_text()):
        caption = str(row.get('caption', '')).strip()
        if caption and row['path'] not in cache:
            cache[row['path']] = caption
            added += 1
    save_cache(cache_path, cache)
    print({'added': added, 'cache_entries': len(cache)})


if __name__ == '__main__':
    main()
