"""Reuse frozen clean signatures for the paired robustness screen."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--screen', default='data/robustness_screen/manifest.csv')
    parser.add_argument('--source', default='results/features/stage_b_cross_source_rich.json')
    parser.add_argument('--output', default='results/robustness_screen/rich_features.json')
    args = parser.parse_args()
    screen = pd.read_csv(args.screen, keep_default_na=False)
    clean = screen.loc[screen['transform'].eq('clean')].copy()
    if len(clean) != 120 or clean['original_id'].nunique() != 120:
        raise ValueError(f'expected 120 unique clean rows, got {len(clean)}')
    by_path = {str(row['path']): row for row in json.loads(Path(args.source).read_text())}
    missing = sorted(set(clean.path) - set(by_path))
    if missing:
        raise ValueError(f'{len(missing)} clean screen images missing from source cache')
    output = Path(args.output)
    existing = json.loads(output.read_text()) if output.exists() else []
    existing_keys = {(str(x['path']), str(x.get('transform', '')), str(x.get('transform_value', '')), str(x.get('extractor_protocol_version', ''))) for x in existing}
    added = 0
    for row in clean.to_dict('records'):
        key = (str(row['path']), 'clean', str(row['transform_value']), str(row['extractor_protocol_version']))
        if key in existing_keys:
            continue
        cached = dict(by_path[str(row['path'])])
        cached.update({'original_id': str(row['original_id']), 'transform': 'clean', 'transform_value': row['transform_value'], 'transform_seed': int(row['transform_seed']), 'extractor_protocol_version': str(row['extractor_protocol_version'])})
        existing.append(cached)
        added += 1
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(existing, indent=2))
    print({'clean_rows': len(clean), 'added': added, 'cache_rows': len(existing)})


if __name__ == '__main__':
    main()
