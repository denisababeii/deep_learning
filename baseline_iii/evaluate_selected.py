#!/usr/bin/env python3
"""Evaluate the selected MLP on the test set for Part 5."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import torch

from finetune import metrics, predict
from models.baselines import FeatureModel


def restore_model(run, data, device='cpu'):
    """Load the saved model, feature scaler and target statistics."""
    run = Path(run)
    config = json.loads((run / 'config.json').read_text())
    cfg = OmegaConf.create(config)
    if cfg.model.type != 'features':
        raise ValueError('This evaluator supports the Part 2 feature MLP only')
    model = FeatureModel(cfg.model, cfg.readout, len(cfg.targets), Path(data), device)
    model.scaler = joblib.load(run / 'feature_scaler.joblib')
    if cfg.model.scaling == 'quantile' and model.scaler is None:
        raise ValueError('Missing fitted training feature scaler')
    saved = torch.load(run / 'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(saved['model'])
    model.eval()
    return model, saved['mean'], saved['std'], cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, default=Path('runs/parts12/selected_models.json'))
    parser.add_argument('--task', choices=['regression', 'classification'], required=True)
    parser.add_argument('--data', type=Path, default=Path('data'))
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    if not selection.get('frozen'):
        raise ValueError('Freeze the validation-selected models before evaluating test')
    selected = selection['models'][args.task]
    run = Path(selected['run_dir'])
    for name, expected in selected['artifact_sha256'].items():
        if hashlib.sha256((run / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Selected model artifact changed: {name}')
    for split, expected in selected['request']['split_sha256'].items():
        if hashlib.sha256((args.data / args.task / f'{split}.csv').read_bytes()).hexdigest() != expected:
            raise ValueError(f'Data split changed: {split}')
    out = args.selection.parent / 'part5' / args.task
    if out.exists():
        raise ValueError(f'Final evaluation already exists: {out}')
    torch.set_num_threads(4)
    model, mean, std, cfg = restore_model(run, args.data)
    # Check that reloading the model reproduces its validation predictions.
    val = pd.read_csv(args.data / args.task / 'val.csv')
    vp = predict(model, model.loader(val, cfg.targets, cfg.train.batch_size), args.task, mean, std)
    expected_predictions = pd.read_csv(run / 'validation_predictions.csv')[list(cfg.targets)].to_numpy()
    np.testing.assert_allclose(vp, expected_predictions, rtol=1e-5, atol=1e-6)
    test = pd.read_csv(args.data / args.task / 'test.csv')
    development_ids = pd.read_csv(args.data / args.task / 'train.csv', usecols=['canonical_smiles'])
    if set(test.canonical_smiles) & (set(val.canonical_smiles) | set(development_ids.canonical_smiles)):
        raise ValueError('Molecule overlap with test')
    predictions = predict(model, model.loader(test, cfg.targets, cfg.train.batch_size), args.task, mean, std)
    scores = metrics(test, predictions, cfg.targets, args.task)
    out.mkdir(parents=True)
    (out / 'test_metrics.json').write_text(json.dumps(scores, indent=2))
    output = test[['SMILES', 'Molecule_Name']].copy()
    output[list(cfg.targets)] = predictions
    output.to_csv(out / 'test_predictions.csv', index=False)
    (out / 'selection.json').write_text(json.dumps(selected, indent=2))
    print(json.dumps(scores, indent=2))


if __name__ == '__main__':
    main()
