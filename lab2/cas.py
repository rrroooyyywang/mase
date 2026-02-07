import os
import math
import gc
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import optuna
from optuna.samplers import RandomSampler, TPESampler, GridSampler
from optuna.trial import TrialState

from transformers import AutoConfig, AutoModelForSequenceClassification
from chop.tools import get_tokenized_dataset, get_trainer
from chop.nn.modules import Identity
from chop.tools.utils import deepsetattr
from chop.pipelines import CompressionPipeline
from chop import MaseGraph
import copy

BEST_SAMPLER = "tpe"

N_TRIALS = 50

CHECKPOINT = "prajjwal1/bert-tiny"
TOKENIZER_CHECKPOINT = "bert-base-uncased"
DATASET_NAME = "imdb"

PRE_COMPRESSION_EPOCHS = 1
POST_COMPRESSION_EPOCHS = 1 

TASK1_RESULT = "/home/roy/Documents/STUDY/IC/YEAR4_TERM2/ADLS/mase/lab2/results_tpe_1.txt"
OUT_TXT_CAS_NO_POST = "results_cas_no_post.txt"
OUT_TXT_CAS_POST = "results_cas_post.txt"
OUT_FIG = "cas_three_curves.png"

SEARCH_SPACE = {
    "num_layers": [2, 4, 8],
    "num_heads": [2, 4, 8, 16],
    "hidden_size": [128, 192, 256, 384, 512],
    "intermediate_size": [512, 768, 1024, 1536, 2048],
}

LINEAR_CHOICES = ["linear", "identity"]

QUANTIZATION_CONFIG = {
    "by": "type",
    "default": {"config": {"name": None}},
    "linear": {
        "config": {
            "name": "integer",
            "data_in_width": 8,
            "data_in_frac_width": 4,
            "weight_width": 8,
            "weight_frac_width": 4,
            "bias_width": 8,
            "bias_frac_width": 4,
        }
    },
}

PRUNING_CONFIG = {
    "weight": {"sparsity": 0.5, "method": "l1-norm", "scope": "local"},
    "activation": {"sparsity": 0.5, "method": "l1-norm", "scope": "local"},
}


def build_sampler(name: str):
    name = name.lower()
    if name == "random":
        return RandomSampler(seed=0)
    if name == "tpe":
        return TPESampler(seed=0, multivariate=True)
    if name == "grid":
        grid_space = {
            "num_layers": list(range(len(SEARCH_SPACE["num_layers"]))),
            "num_heads": list(range(len(SEARCH_SPACE["num_heads"]))),
            "hidden_size": list(range(len(SEARCH_SPACE["hidden_size"]))),
            "intermediate_size": list(range(len(SEARCH_SPACE["intermediate_size"]))),
        }
        return GridSampler(grid_space)
    raise ValueError(f"Unknown sampler: {name}")


def export_trials_to_txt(study: optuna.Study, filepath: str):
    """Write: trial_number trial_accuracy best_so_far_accuracy"""
    trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    trials.sort(key=lambda t: t.number)

    best = -math.inf if study.direction.name == "MAXIMIZE" else math.inf

    with open(filepath, "w") as f:
        f.write("# trial_number trial_accuracy best_so_far_accuracy\n")
        for t in trials:
            v = t.value
            if v is None:
                continue
            if study.direction.name == "MAXIMIZE":
                best = max(best, v)
            else:
                best = min(best, v)
            f.write(f"{t.number} {v:.6f} {best:.6f}\n")


def load_best_curve_from_txt(path: str):
    """Return x=[1..N], y=best_so_far from exported txt."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing txt file: {path}")
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    best = data[:, 2]
    x = np.arange(1, len(best) + 1)
    return x, best


def construct_model(trial: optuna.Trial):
    config = AutoConfig.from_pretrained(CHECKPOINT)

    for param in ["num_layers", "num_heads", "hidden_size", "intermediate_size"]:
        idx = trial.suggest_int(param, 0, len(SEARCH_SPACE[param]) - 1)
        setattr(config, param, SEARCH_SPACE[param][idx])

    model = AutoModelForSequenceClassification.from_config(config)

    for name, layer in model.named_modules():
        if isinstance(layer, nn.Linear) and layer.in_features == layer.out_features:
            choice = trial.suggest_categorical(f"{name}_type", LINEAR_CHOICES)
            if choice == "identity":
                deepsetattr(model, name, Identity())

    return model


def make_objective(post_compression_epochs: int):
    def objective(trial: optuna.Trial):
        model = construct_model(trial)

        trainer_pre = get_trainer(
            model=model,
            tokenized_dataset=dataset,
            tokenizer=tokenizer,
            evaluate_metric="accuracy",
            num_train_epochs=PRE_COMPRESSION_EPOCHS,
        )
        trainer_pre.train()
        model = trainer_pre.model
        model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mg = MaseGraph(
            model,
            hf_input_names=[
                "input_ids",
                "attention_mask",
                "labels",
            ],
        )
        pipe = CompressionPipeline()
        qcfg = copy.deepcopy(QUANTIZATION_CONFIG)
        pcfg = copy.deepcopy(PRUNING_CONFIG)
        mg, _ = pipe(
            mg,
            pass_args={
                "quantize_transform_pass": qcfg,
                "prune_transform_pass": pcfg,
            },
        )
        compressed_model = mg.model

        if post_compression_epochs > 0:
            trainer_post = get_trainer(
                model=compressed_model,
                tokenized_dataset=dataset,
                tokenizer=tokenizer,
                evaluate_metric="accuracy",
                num_train_epochs=post_compression_epochs,
            )
            trainer_post.train()
            eval_results = trainer_post.evaluate()
        else:
            trainer_eval = get_trainer(
                model=compressed_model,
                tokenized_dataset=dataset,
                tokenizer=tokenizer,
                evaluate_metric="accuracy",
                num_train_epochs=0,
            )
            eval_results = trainer_eval.evaluate()

        acc = float(eval_results["eval_accuracy"])

        del model, trainer_pre
        try:
            del trainer_post
        except Exception:
            pass
        try:
            del trainer_eval
        except Exception:
            pass
        del mg, compressed_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return acc

    return objective



if __name__ == "__main__":
    dataset, tokenizer = get_tokenized_dataset(
        dataset=DATASET_NAME,
        checkpoint=TOKENIZER_CHECKPOINT,
        return_tokenizer=True,
    )

    sampler = build_sampler(BEST_SAMPLER)

    study_no_post = optuna.create_study(
        direction="maximize",
        study_name=f"cas-no-post-{BEST_SAMPLER}",
        sampler=sampler,
    )
    study_no_post.optimize(
        make_objective(post_compression_epochs=0),
        n_trials=N_TRIALS,
        n_jobs=1,
        show_progress_bar=True,
        catch=(Exception,),
    )
    export_trials_to_txt(study_no_post, OUT_TXT_CAS_NO_POST)
    print(f"[INFO] Wrote: {OUT_TXT_CAS_NO_POST}")

    study_post = optuna.create_study(
        direction="maximize",
        study_name=f"cas-post-{BEST_SAMPLER}",
        sampler=build_sampler(BEST_SAMPLER),
    )
    study_post.optimize(
        make_objective(post_compression_epochs=POST_COMPRESSION_EPOCHS),
        n_trials=N_TRIALS,
        n_jobs=1,
        show_progress_bar=True,
        catch=(Exception,),
    )
    export_trials_to_txt(study_post, OUT_TXT_CAS_POST)
    print(f"[INFO] Wrote: {OUT_TXT_CAS_POST}")

    x1, y1 = load_best_curve_from_txt(TASK1_RESULT)
    x2, y2 = load_best_curve_from_txt(OUT_TXT_CAS_NO_POST)
    x3, y3 = load_best_curve_from_txt(OUT_TXT_CAS_POST)

    plt.figure(figsize=(7.5, 5.2))
    plt.plot(x1, y1, marker="o", label="Task 1 (NAS, no compression)")
    plt.plot(x2, y2, marker="o", label="CAS (no post-compression training)")
    plt.plot(x3, y3, marker="o", label="CAS (with post-compression training)")

    plt.title("Best-so-far Accuracy vs Number of Trials")
    plt.xlabel("Number of Trials")
    plt.ylabel("Maximum Achieved Accuracy (Best-so-far)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=200)
    plt.show()
    print(f"[INFO] Saved plot: {OUT_FIG}")
