from pathlib import Path
import dill
from chop.tools import get_tokenized_dataset
from networkx import config
from pytest import param
import torch
from chop.tools.utils import deepsetattr
from copy import deepcopy
from chop.tools import get_trainer
import random
from optuna.samplers import GridSampler, RandomSampler, TPESampler
import optuna
from chop.nn.quantized.modules.linear import (
    LinearInteger,
    LinearMinifloatDenorm,
    LinearMinifloatIEEE,
    LinearLog,
    LinearBlockFP,
    LinearBlockMinifloat,
    LinearBlockLog,
    LinearBinary,
    LinearBinaryScaling,
    LinearBinaryResidualSign,
)
import matplotlib.pyplot as plt

def quant_config(int_width=8, int_frac_width=4):
    return {
        "by": "type",
        "default": {
            "config": {
                "name": None,
            }
        },
        "linear": {
            "config": {
                "name": "integer",
                # data
                "data_in_width": int_width,
                "data_in_frac_width": int_frac_width,
                # weight
                "weight_width": int_width,
                "weight_frac_width": int_frac_width,
                # bias
                "bias_width": int_width,
                "bias_frac_width": int_frac_width,
            }
        },
    }


def construct_model(trial):
    trial_model = deepcopy(base_model)

    for name, layer in trial_model.named_modules():
        if not isinstance(layer, torch.nn.Linear):
            continue

        new_layer_cls = trial.suggest_categorical(
            f"{name}_type",
            search_space["linear_layer_choices"],
        )

        if new_layer_cls == torch.nn.Linear:
            continue

        kwargs = {
            "in_features": layer.in_features,
            "out_features": layer.out_features,
        }

        if new_layer_cls == LinearInteger:
            w = trial.suggest_categorical(f"{name}_int_width", search_space["quantization_width_choices"])
            f = trial.suggest_categorical(f"{name}_int_frac_width", search_space["quantization_frac_width_choices"])

            kwargs["config"] = {
                "data_in_width": w,
                "data_in_frac_width": f,
                "weight_width": w,
                "weight_frac_width": f,
                "bias_width": w,
                "bias_frac_width": f,
            }

        new_layer = new_layer_cls(**kwargs)
        new_layer.weight.data = layer.weight.data
        if getattr(layer, "bias", None) is not None and getattr(new_layer, "bias", None) is not None:
            new_layer.bias.data = layer.bias.data

        deepsetattr(trial_model, name, new_layer)

    return trial_model

def objective(trial):

    # Define the model
    model = construct_model(trial)

    trainer = get_trainer(
        model=model,
        tokenized_dataset=dataset,
        tokenizer=tokenizer,
        evaluate_metric="accuracy",
        num_train_epochs=1,
    )

    trainer.train()
    eval_results = trainer.evaluate()

    trial.set_user_attr("model", model)

    return eval_results["eval_accuracy"]

if __name__ == "__main__":
    TRAILS = 50
    checkpoint = "prajjwal1/bert-tiny"
    tokenizer_checkpoint = "bert-base-uncased"
    dataset_name = "imdb"

    with open(f"/home/roy/Documents/STUDY/IC/YEAR4_TERM2/ADLS/mase/lab2/tutorial_5_best_model.pkl", "rb") as f:
        base_model = dill.load(f)

    dataset, tokenizer = get_tokenized_dataset(
        dataset=dataset_name,
        checkpoint=tokenizer_checkpoint,
        return_tokenizer=True,
    )

    search_space = {
        "quantization_width_choices": [
            8, 16, 32
        ],
        "quantization_frac_width_choices": [
            2, 4, 8
        ],
        "linear_layer_choices": [
            torch.nn.Linear,
            LinearInteger,
        ],
    }

    sampler = TPESampler()

    study = optuna.create_study(
        direction="maximize",
        study_name="bert-tiny-nas-study",
        sampler=sampler,
    )

    study.optimize(
        objective,
        n_trials=TRAILS,
        timeout=60 * 60 * 24,
    )

    vals = [t.value for t in study.trials if t.value is not None]

    best_so_far = []
    cur = float("-inf")
    for v in vals:
        cur = max(cur, v)
        best_so_far.append(cur)

    plt.figure()
    plt.plot(range(1, len(best_so_far) + 1), best_so_far, marker="o")
    plt.xlabel("Number of trials")
    plt.ylabel("Best accuracy so far")
    plt.title("Mixed Precision Tuning: accuracy vs trials")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.savefig("mixed_precision_tuning.png", dpi=200)