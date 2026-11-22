
import json
import subprocess
import sys
from pathlib import Path

import optuna

ROOT = Path(__file__).resolve().parent
TASK = sys.argv[1]
N_TRIALS = 25

def objective(trial):
    hidden_dim = trial.suggest_categorical(
        "hidden_dim", [128, 256, 512]
    )
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.0, 0.3)

    readout_lr = trial.suggest_float(
        "readout_lr", 1e-4, 1e-2, log=True
    )
    weight_decay = trial.suggest_float(
        "weight_decay", 1e-6, 1e-3, log=True
    )

    epochs = trial.suggest_categorical("epochs", [30, 50, 70, 100])
    freeze_epochs = trial.suggest_int("freeze_epochs", 5, 20)
    run_dir = ROOT / "optuna_runs" / f"{TASK}_trial_{trial.number}"
    run_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable, "finetune.py",
        "model=atom_jepa",
        f"task={TASK}",
        "model.freeze_encoder=true",
        f"model.readout_lr={readout_lr}",
        f"train.weight_decay={weight_decay}",
        f"readout.hidden_dim={hidden_dim}",
        f"readout.num_layers={num_layers}",
        f"readout.dropout={dropout}",
        f"train.epochs={epochs}",
        "evaluate_test=false",
        f"hydra.run.dir={run_dir}",
    ]

    subprocess.run(command, cwd=ROOT, check=True)

    with open(run_dir / "validation_metrics.json") as f:
        metrics = json.load(f)

    if TASK == "regression":
        return metrics["macro_ST-RAE"]

    return metrics["macro_MCC"]


direction = "minimize" if TASK == "regression" else "maximize"

study = optuna.create_study(
    direction=direction,
    study_name=f"atom_jepa_{TASK}",
)

study.optimize(objective, n_trials=N_TRIALS)

study.trials_dataframe().to_csv(
    ROOT / f"optuna_{TASK}_results.csv",
    index=False
)

