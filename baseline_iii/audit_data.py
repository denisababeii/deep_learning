#!/usr/bin/env python3
"""Check the data splits, labels and conformers."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_data import TASK_LABELS


def audit_data(data, out):
    data, out = Path(data), Path(out)
    manifest = json.loads((data / 'manifest.json').read_text())
    settings = manifest['config']
    if (settings['seed'], settings['val_fraction'], settings['test_fraction'], settings['limit']) != (42, .15, .15, None):
        raise ValueError('Experiments require complete starter data with seed 42 and default fractions')
    if 'counts' not in manifest:
        raise ValueError('Preparation did not finish')
    identities, counts, summaries, hashes = [], [], [], {}
    identity_columns = ['Molecule_Name', 'canonical_smiles', 'key', 'split']
    for task, targets in TASK_LABELS.items():
        hashes[task] = {}
        for split in ['train', 'val', 'test']:
            path = data / task / f'{split}.csv'
            hashes[task][split] = hashlib.sha256(path.read_bytes()).hexdigest()
            # Read molecule IDs to check overlap between splits.
            ids = pd.read_csv(path, usecols=identity_columns)
            if ids.empty or ids.isna().any().any() or not ids.Molecule_Name.is_unique:
                raise ValueError(f'Empty or invalid identities in {path}')
            if not ids.split.eq(split).all():
                raise ValueError(f'Incorrect split column in {path}')
            if len(ids) != manifest['counts'][task][split]:
                raise ValueError(f'Split count disagrees with preparation manifest: {path}')
            identities.append(ids)
            counts.append({'task': task, 'split': split, 'rows': len(ids),
                           'canonical_molecules': ids.canonical_smiles.nunique()})
            if split == 'test':
                continue
            frame = pd.read_csv(path)
            if not frame[targets].notna().any(axis=1).all():
                raise ValueError(f'Rows without any task labels in {path}')
            for target in targets:
                y = frame[target].dropna()
                if y.empty or not np.isfinite(y).all():
                    raise ValueError(f'Missing or nonfinite endpoint: {path}, {target}')
                row = {'task': task, 'split': split, 'target': target,
                       'observed': len(y), 'missing': int(frame[target].isna().sum())}
                if task == 'classification':
                    if set(y.unique()) != {0, 1}:
                        raise ValueError(f'Both binary classes required for {path}, {target}')
                    row.update(positives=int(y.sum()), negatives=int((y == 0).sum()),
                               positive_fraction=float(y.mean()))
                else:
                    intervals = frame.loc[y.index, [target + '_conf_low', target + '_conf_high']]
                    if not np.isfinite(intervals).all().all() or not intervals.iloc[:, 0].le(intervals.iloc[:, 1]).all():
                        raise ValueError(f'Invalid uncertainty intervals: {path}, {target}')
                    row.update({k: float(v) for k, v in y.describe().items() if k != 'count'})
                summaries.append(row)
    ids = pd.concat(identities, ignore_index=True)
    for column in ['canonical_smiles', 'Molecule_Name', 'key']:
        if ids.groupby(column).split.nunique().gt(1).any():
            raise ValueError(f'Molecule leakage across tasks or splits by {column}')
    unique_keys = ids.key.unique()
    for key in unique_keys:
        with np.load(data / 'conformers' / f'{key}.npz') as sample:
            z, pos = sample['z'], sample['pos']
            if z.ndim != 1 or pos.shape != (len(z), 3) or not len(z) or not (z > 0).all() or not np.isfinite(pos).all():
                raise ValueError(f'Invalid conformer: {key}')
    conformer_report = pd.read_csv(data / 'conformer_report.csv', keep_default_na=False)
    failures = conformer_report[conformer_report.error.ne('')]
    if len(failures) != manifest['failed_conformers']:
        raise ValueError('Conformer failure report disagrees with manifest')
    if set(failures.key) & set(unique_keys):
        raise ValueError('Failed conformer remained in the prepared splits')
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(counts).to_csv(out / 'split_counts.csv', index=False)
    pd.DataFrame(summaries).to_csv(out / 'label_summary.csv', index=False)
    report = {'passed': True, 'preparation': manifest, 'split_sha256': hashes,
              'canonical_molecules': int(ids.canonical_smiles.nunique()),
              'validated_conformers': len(unique_keys), 'test_labels_read': False,
              'conformer_failures': failures.to_dict('records')}
    (out / 'audit.json').write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--out', type=Path, default=Path('runs/parts12-audit'))
    args = parser.parse_args()
    report = audit_data(args.data, args.out)
    print(json.dumps({k: report[k] for k in ['passed', 'canonical_molecules', 'validated_conformers', 'test_labels_read']}, indent=2))
