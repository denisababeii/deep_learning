#!/usr/bin/env python3
"""Train and compare Morgan + RDKit MLP settings on CPU."""
import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from baseline_iii.audit_data import audit_data
from prepare_data import TASK_LABELS

PROJECT = Path(__file__).resolve().parent.parent
EXPERIMENTS = [
    ('default', {}),
    ('lower_lr', {'model.readout_lr': .0003}),
    ('higher_lr', {'model.readout_lr': .003}),
    ('narrower', {'readout.hidden_dim': 256}),      # fewer units per hidden layer
    ('shallower', {'readout.num_layers': 2}),       # one hidden layer
    ('more_dropout', {'readout.dropout': .3}),      # more dropout during training
    ('less_weight_decay', {'train.weight_decay': .001}),
    ('larger_batch', {'train.batch_size': 128}),
]
DEFAULTS = {'model.readout_lr': .001, 'readout.hidden_dim': 512,
            'readout.num_layers': 3, 'readout.dropout': .1,
            'train.weight_decay': .01, 'train.batch_size': 64}


def primary(task):
    return 'macro_ST-RAE' if task == 'regression' else 'macro_MCC'


def rank(record):
    score = record['metrics'][primary(record['task'])]
    return score if record['task'] == 'regression' else -score


def combined_settings(records):
    """Combine the best setting from each comparison. Keep defaults on ties."""
    baseline = next(r for r in records if r['experiment'] == 'default')
    settings = {}
    for key, default in DEFAULTS.items():
        candidates = [baseline] + [r for r in records
                                   if key in r['overrides'] and len(r['overrides']) == 1]
        winner = min(candidates, key=rank)
        value = winner['overrides'].get(key, default)
        if value != default:
            settings[key] = value
    return settings


def source_hash():
    paths = [PROJECT / 'finetune.py', PROJECT / 'prepare_data.py', PROJECT / 'requirements.txt']
    paths += sorted((PROJECT / 'conf').rglob('*.yaml'))
    paths += sorted((PROJECT / 'models').glob('*.py'))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(PROJECT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    # Finish writing before replacing the existing file.
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def run_training(command, env, run_dir, task, epochs):
    """Show epoch progress and save the full training output."""
    started = time.perf_counter()
    with (run_dir / 'console.log').open('w') as log:
        with subprocess.Popen(command, cwd=PROJECT, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        print(line, end='', flush=True)
                        continue
                    if isinstance(record, dict) and 'epoch' in record and 'train_loss' in record:
                        elapsed = time.perf_counter() - started
                        metric = primary(task)
                        print(f'Epoch {record["epoch"]}/{epochs} | '
                              f'loss {record["train_loss"]:.4f} | '
                              f'{metric} {record[metric]:.6f} | elapsed {elapsed:.1f}s',
                              flush=True)
                    else:
                        print(line, end='', flush=True)
                returncode = process.wait()
            except KeyboardInterrupt:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
    return returncode, time.perf_counter() - started


def run_one(task, name, overrides, epochs, data, root, audit):
    folder = root / task / name
    folder.mkdir(parents=True, exist_ok=True)
    expected = {'task': task, 'experiment': name, 'overrides': overrides, 'epochs': epochs,
                'source_sha256': source_hash(), 'split_sha256': audit['split_sha256'][task],
                'data_dir': str(data), 'device': 'cpu', 'seed': 42,
                'versions': {p: importlib.metadata.version(p) for p in
                             ['torch', 'numpy', 'pandas', 'scikit-learn', 'rdkit', 'descriptastorus']}}
    marker = folder / 'complete.json'
    if marker.exists():
        record = read_json(marker)
        if record['request'] != expected:
            raise ValueError(f'Completed run settings changed: {folder}. Use a new --out directory.')
        for name_, digest in record['artifact_sha256'].items():
            if file_hash(Path(record['run_dir']) / name_) != digest:
                raise ValueError(f'Completed artifact changed: {folder}, {name_}')
        if list(Path(record['run_dir']).glob('test_*')):
            raise ValueError(f'Test artifacts present in validation run: {folder}')
        print(f'Reusing {task}/{name}', flush=True)
        return record
    run_dir = folder / datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
    run_dir.mkdir()
    command = [sys.executable, '-u', str(PROJECT / 'finetune.py'),
               'model=morgan_rdkit_mlp', f'task={task}', 'device=cpu', 'seed=42',
               'evaluate_test=false', f'train.epochs={epochs}', f'data={data}',
               f'hydra.run.dir={run_dir}']
    command += [f'{key}={value}' for key, value in overrides.items()]
    env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    write_json(root / 'progress.json', {'task': task, 'experiment': name,
                                       'run_dir': str(run_dir), 'status': 'running'})
    print(f'Training {task}/{name}: {run_dir}', flush=True)
    returncode, elapsed = run_training(command, env, run_dir, task, epochs)
    if returncode:
        write_json(root / 'progress.json', {'task': task, 'experiment': name,
                                           'run_dir': str(run_dir), 'status': 'failed'})
        raise RuntimeError(f'{task}/{name} failed; see {run_dir / "console.log"}')
    if list(run_dir.glob('test_*')):
        raise RuntimeError('Validation-only run unexpectedly produced test artifacts')
    history = [json.loads(line) for line in (run_dir / 'history.jsonl').read_text().splitlines()]
    if len(history) != epochs or not all(np.isfinite(r['train_loss']) for r in history):
        raise ValueError(f'Incomplete history or nonfinite losses in {run_dir}')
    scores = read_json(run_dir / 'validation_metrics.json')
    if scores[primary(task)] is None or not np.isfinite(scores[primary(task)]):
        raise ValueError(f'Undefined primary score in {run_dir}')
    selection = read_json(run_dir / 'selection.json')
    record = {'request': expected, 'task': task, 'experiment': name, 'overrides': overrides,
              'run_dir': str(run_dir), 'metrics': scores, **selection,
              **read_json(run_dir / 'runtime.json')}
    artifacts = ['config.json', 'best.pt', 'feature_scaler.joblib', 'history.jsonl',
                 'validation_metrics.json', 'validation_predictions.csv', 'selection.json', 'runtime.json']
    record['artifact_sha256'] = {name_: file_hash(run_dir / name_) for name_ in artifacts}
    write_json(marker, record)
    print(f'Finished {task}/{name}: {primary(task)}={scores[primary(task)]:.6f} | '
          f'time {elapsed:.1f}s ({elapsed / 60:.2f} min)', flush=True)
    return record


def records_at(root):
    records = []
    for task in TASK_LABELS:
        for name in [name for name, _ in EXPERIMENTS] + ['combined']:
            marker = root / task / name / 'complete.json'
            if marker.exists():
                records.append(read_json(marker))
    return records


def report(root, finalize=False):
    records = records_at(root)
    if not records:
        return
    rows = []
    for record in records:
        task = record['task']
        baseline = next(r for r in records if r['task'] == task and r['experiment'] == 'default')
        row = {k: record[k] for k in ['task', 'experiment', 'best_epoch', 'validation_runtime_seconds', 'run_dir']}
        row.update({k: DEFAULTS[k] for k in DEFAULTS})
        row.update(record['overrides'])
        row[primary(task)] = record['metrics'][primary(task)]
        row['improvement_over_default'] = rank(baseline) - rank(record)
        for target in TASK_LABELS[task]:
            row[target] = record['metrics'][target]['ST-RAE' if task == 'regression' else 'MCC']
        rows.append(row)
    pd.DataFrame(rows).to_csv(root / 'summary.csv', index=False)
    # Save the plotting cache with the run outputs.
    os.environ.setdefault('MPLCONFIGDIR', str(root / '.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plot_dir = root / 'plots'
    plot_dir.mkdir(exist_ok=True)
    selections = {}
    enzyme_comparisons = []
    for task, targets in TASK_LABELS.items():
        runs = [r for r in records if r['task'] == task]
        if not runs:
            continue
        base = next(r for r in runs if r['experiment'] == 'default')
        winner = min(runs, key=rank)
        key = primary(task)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for record in runs:
            history = pd.read_json(Path(record['run_dir']) / 'history.jsonl', lines=True)
            axes[0].plot(history.epoch, history.train_loss, label=record['experiment'])
            axes[1].plot(history.epoch, history[key], label=record['experiment'])
        axes[0].set(xlabel='Epoch', ylabel='Training loss', title=f'{task.capitalize()}: optimization')
        axes[1].set(xlabel='Epoch', ylabel=key, title=f'{task.capitalize()}: validation')
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_dir / f'{task}_learning_curves.png', dpi=160)
        plt.close(fig)
        endpoint = 'ST-RAE' if task == 'regression' else 'MCC'
        for target in targets:
            a, b = base['metrics'][target][endpoint], winner['metrics'][target][endpoint]
            gain = a - b if task == 'regression' else b - a
            enzyme_comparisons.append({
                'task': task, 'target': target, 'metric': endpoint,
                'selected_experiment': winner['experiment'],
                'default_score': a, 'selected_score': b, 'improvement_over_default': gain,
            })
        selections[task] = winner
    pd.DataFrame(enzyme_comparisons).to_csv(root / 'per_enzyme_comparison.csv', index=False)
    if finalize:
        for task in TASK_LABELS:
            if len([r for r in records if r['task'] == task]) != 9:
                raise ValueError(f'Cannot freeze {task}: expected all nine full configurations')
        # Reload each selected model and compare its validation predictions.
        from baseline_iii.evaluate_selected import restore_model
        from finetune import predict
        import torch
        torch.set_num_threads(4)
        replay_checks = {}
        for task, selected in selections.items():
            data = Path(selected['request']['data_dir'])
            run = Path(selected['run_dir'])
            model, mean, std, cfg = restore_model(run, data)
            val = pd.read_csv(data / task / 'val.csv')
            restored = predict(model, model.loader(val, cfg.targets, cfg.train.batch_size), task, mean, std)
            original = pd.read_csv(run / 'validation_predictions.csv')[list(cfg.targets)].to_numpy()
            np.testing.assert_allclose(restored, original, rtol=1e-5, atol=1e-6)
            replay_checks[task] = {'passed': True, 'max_abs_error': float(np.max(np.abs(restored - original)))}
        write_json(root / 'validation_replay.json', replay_checks)
        write_json(root / 'selected_models.json', {'frozen': True, 'test_evaluated': False, 'models': selections})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=PROJECT / 'data')
    parser.add_argument('--out', type=Path, default=PROJECT / 'runs/parts12')
    parser.add_argument('--stage', choices=['smoke', 'default', 'all', 'report'], default='all')
    args = parser.parse_args()
    data, root = args.data.resolve(), args.out.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.stage == 'report':
        report(root, finalize=all((root / task / 'combined/complete.json').exists() for task in TASK_LABELS))
        return
    audit = audit_data(data, root / 'audit')
    for task in TASK_LABELS:
        run_one(task, 'smoke', {}, 2, data, root, audit)
    if args.stage == 'smoke':
        write_json(root / 'progress.json', {'status': 'smoke_complete'})
        return
    # Run the default for both tasks before changing settings.
    for task in TASK_LABELS:
        run_one(task, 'default', {}, 100, data, root, audit)
        report(root)
    if args.stage == 'all':
        for task in TASK_LABELS:
            runs = [read_json(root / task / 'default/complete.json')]
            for name, overrides in EXPERIMENTS[1:]:
                runs.append(run_one(task, name, overrides, 100, data, root, audit))
                report(root)
            run_one(task, 'combined', combined_settings(runs), 100, data, root, audit)
            report(root)
        report(root, finalize=True)
    write_json(root / 'progress.json', {'status': f'{args.stage}_complete'})


if __name__ == '__main__':
    main()
