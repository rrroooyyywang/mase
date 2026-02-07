from pathlib import Path
from unicodedata import name
from chop.actions.search import search_space
import dill
import inspect
import torch
from copy import deepcopy
import optuna
from optuna.samplers import TPESampler
import matplotlib.pyplot as plt

from chop.tools import get_tokenized_dataset, get_trainer
from chop.tools.utils import deepsetattr

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

PRECISION_CLASSES = {
    "fp32": torch.nn.Linear,
    "integer": LinearInteger,
    "minifloat_ieee": LinearMinifloatIEEE,
    "minifloat_denorm": LinearMinifloatDenorm,
    "log": LinearLog,
    "block_fp": LinearBlockFP,
    "block_minifloat": LinearBlockMinifloat,
    "block_log": LinearBlockLog,
    "binary": LinearBinary,
    "binary_scaling": LinearBinaryScaling,
    "binary_residual": LinearBinaryResidualSign,
}

SEARCH_SPACE = {
    # Integer
    "int_width": [8, 16, 32],
    "int_frac_width": [2, 4, 8],

    # Minifloat IEEE/Denorm
    "mf_width": [8, 16],
    "mf_exp_width": [3, 4, 5],

    # Log
    "log_width": [8, 16],
    "log_exp_bias": [7, 15, 31],

    # BlockFP
    "bfp_width": [8, 16],
    "bfp_exp_width": [3, 4, 5],
    "bfp_block_size": [8, 16, 32],

    # BlockMinifloat
    "bm_width": [8, 16],
    "bm_exp_width": [3, 4, 5],
    "bm_exp_bias_width": [2, 3, 4],
    "bm_block_size": [8, 16, 32],

    # BlockLog
    "blog_width": [8, 16],
    "blog_exp_bias_width": [2, 3, 4],
    "blog_block_size": [8, 16, 32],

    # Binary
    "bin_weight_stochastic": [False, True],
    "bin_weight_bipolar": [False, True],

    #BinaryScaling
    "bin_data_in_stochastic": [False, True],
    "bin_bias_stochastic": [False, True],

    "bin_data_in_bipolar": [False, True],
    "bin_bias_bipolar": [False, True],

    "bin_binary_training": [False, True],

    # BinaryResidualSign specific
    "binrs_bypass": [False, True],
    "binrs_levels": [2],
    "binrs_residual_sign": [False, True],

    "linear_layer_choices": [
        torch.nn.Linear,
        LinearInteger,
        LinearMinifloatIEEE,
        LinearMinifloatDenorm,
        LinearLog,
        LinearBlockFP,
        LinearBlockMinifloat,
        LinearBlockLog,
        LinearBinary,
        LinearBinaryScaling,
        LinearBinaryResidualSign,
    ],
}



def filter_kwargs_by_signature(cls, kwargs):
    sig = inspect.signature(cls.__init__)
    valid = set(sig.parameters.keys())
    valid.discard("self")
    return {k: v for k, v in kwargs.items() if k in valid}

def build_layer_kwargs(trial, layer_name: str, layer_cls, in_features: int, out_features: int):
    kwargs = {"in_features": in_features, "out_features": out_features}

    if layer_cls is torch.nn.Linear:
        return filter_kwargs_by_signature(layer_cls, kwargs)

    if layer_cls is LinearInteger:
        w = trial.suggest_categorical(f"{layer_name}_int_width", SEARCH_SPACE["int_width"])
        f = trial.suggest_categorical(f"{layer_name}_int_frac_width", SEARCH_SPACE["int_frac_width"])
        kwargs["config"] = {
            "data_in_width": w, "data_in_frac_width": f,
            "weight_width": w,  "weight_frac_width": f,
            "bias_width": w,    "bias_frac_width": f,
        }
        return filter_kwargs_by_signature(layer_cls, kwargs)

    if layer_cls in (LinearMinifloatIEEE, LinearMinifloatDenorm):
        w_width = trial.suggest_categorical(f"{layer_name}_w_mf_width", SEARCH_SPACE["mf_width"])
        x_width = trial.suggest_categorical(f"{layer_name}_x_mf_width", SEARCH_SPACE["mf_width"])
        b_width = trial.suggest_categorical(f"{layer_name}_b_mf_width", SEARCH_SPACE["mf_width"])

        w_exp = trial.suggest_categorical(f"{layer_name}_w_mf_exp_width", SEARCH_SPACE["mf_exp_width"])
        x_exp = trial.suggest_categorical(f"{layer_name}_x_mf_exp_width", SEARCH_SPACE["mf_exp_width"])
        b_exp = trial.suggest_categorical(f"{layer_name}_b_mf_exp_width", SEARCH_SPACE["mf_exp_width"])

        w_bias = (2 ** (w_exp - 1)) - 1
        x_bias = (2 ** (x_exp - 1)) - 1
        b_bias = (2 ** (b_exp - 1)) - 1

        kwargs["config"] = {
            "weight_width": w_width,
            "weight_exponent_width": w_exp,
            "weight_exponent_bias": w_bias,

            "data_in_width": x_width,
            "data_in_exponent_width": x_exp,
            "data_in_exponent_bias": x_bias,

            "bias_width": b_width,
            "bias_exponent_width": b_exp,
            "bias_exponent_bias": b_bias,
        }
        return filter_kwargs_by_signature(layer_cls, kwargs)

    if layer_cls is LinearLog:
        w_width = trial.suggest_categorical(f"{layer_name}_w_log_width", SEARCH_SPACE["log_width"])
        x_width = trial.suggest_categorical(f"{layer_name}_x_log_width", SEARCH_SPACE["log_width"])
        b_width = trial.suggest_categorical(f"{layer_name}_b_log_width", SEARCH_SPACE["log_width"])

        w_bias = trial.suggest_categorical(f"{layer_name}_w_log_exp_bias", SEARCH_SPACE["log_exp_bias"])
        x_bias = trial.suggest_categorical(f"{layer_name}_x_log_exp_bias", SEARCH_SPACE["log_exp_bias"])
        b_bias = trial.suggest_categorical(f"{layer_name}_b_log_exp_bias", SEARCH_SPACE["log_exp_bias"])

        kwargs["config"] = {
            "weight_width": w_width,
            "weight_exponent_bias": w_bias,

            "data_in_width": x_width,
            "data_in_exponent_bias": x_bias,

            "bias_width": b_width,
            "bias_exponent_bias": b_bias,
        }
        return filter_kwargs_by_signature(layer_cls, kwargs)

    if layer_cls is LinearBlockFP:
        w_width = trial.suggest_categorical(f"{layer_name}_w_bfp_width", SEARCH_SPACE["bfp_width"])
        x_width = trial.suggest_categorical(f"{layer_name}_x_bfp_width", SEARCH_SPACE["bfp_width"])
        b_width = trial.suggest_categorical(f"{layer_name}_b_bfp_width", SEARCH_SPACE["bfp_width"])

        w_exp = trial.suggest_categorical(f"{layer_name}_w_bfp_exp_width", SEARCH_SPACE["bfp_exp_width"])
        x_exp = trial.suggest_categorical(f"{layer_name}_x_bfp_exp_width", SEARCH_SPACE["bfp_exp_width"])
        b_exp = trial.suggest_categorical(f"{layer_name}_b_bfp_exp_width", SEARCH_SPACE["bfp_exp_width"])

        w_bias = (2 ** (w_exp - 1)) - 1
        x_bias = (2 ** (x_exp - 1)) - 1
        b_bias = (2 ** (b_exp - 1)) - 1

        w_bs = trial.suggest_categorical(f"{layer_name}_w_bfp_block", SEARCH_SPACE["bfp_block_size"])
        x_bs = trial.suggest_categorical(f"{layer_name}_x_bfp_block", SEARCH_SPACE["bfp_block_size"])
        b_bs = trial.suggest_categorical(f"{layer_name}_b_bfp_block", SEARCH_SPACE["bfp_block_size"])

        x_skip = True

        kwargs["config"] = {
            # weight
            "weight_width": w_width,
            "weight_exponent_width": w_exp,
            "weight_exponent_bias": w_bias,
            "weight_block_size": w_bs,

            # data_in
            "data_in_width": x_width,
            "data_in_exponent_width": x_exp,
            "data_in_exponent_bias": x_bias,
            "data_in_block_size": x_bs,
            "data_in_skip_first_dim": x_skip,

            # bias
            "bias_width": b_width,
            "bias_exponent_width": b_exp,
            "bias_exponent_bias": b_bias,
            "bias_block_size": b_bs,
        }
        return filter_kwargs_by_signature(layer_cls, kwargs)
    
    if layer_cls is LinearBlockMinifloat:

        w_width = trial.suggest_categorical(f"{layer_name}_w_bm_width", SEARCH_SPACE["bm_width"])
        x_width = trial.suggest_categorical(f"{layer_name}_x_bm_width", SEARCH_SPACE["bm_width"])
        b_width = trial.suggest_categorical(f"{layer_name}_b_bm_width", SEARCH_SPACE["bm_width"])

        w_exp = trial.suggest_categorical(f"{layer_name}_w_bm_exp_width", SEARCH_SPACE["bm_exp_width"])
        x_exp = trial.suggest_categorical(f"{layer_name}_x_bm_exp_width", SEARCH_SPACE["bm_exp_width"])
        b_exp = trial.suggest_categorical(f"{layer_name}_b_bm_exp_width", SEARCH_SPACE["bm_exp_width"])

        w_ebw = trial.suggest_categorical(f"{layer_name}_w_bm_exp_bias_width", SEARCH_SPACE["bm_exp_bias_width"])
        x_ebw = trial.suggest_categorical(f"{layer_name}_x_bm_exp_bias_width", SEARCH_SPACE["bm_exp_bias_width"])
        b_ebw = trial.suggest_categorical(f"{layer_name}_b_bm_exp_bias_width", SEARCH_SPACE["bm_exp_bias_width"])

        w_bs = trial.suggest_categorical(f"{layer_name}_w_bm_block", SEARCH_SPACE["bm_block_size"])
        x_bs = trial.suggest_categorical(f"{layer_name}_x_bm_block", SEARCH_SPACE["bm_block_size"])
        b_bs = trial.suggest_categorical(f"{layer_name}_b_bm_block", SEARCH_SPACE["bm_block_size"])

        x_skip = True

        kwargs["config"] = {
            "weight_width": w_width,
            "weight_exponent_width": w_exp,
            "weight_exponent_bias_width": w_ebw,
            "weight_block_size": [w_bs],

            "data_in_width": x_width,
            "data_in_exponent_width": x_exp,
            "data_in_exponent_bias_width": x_ebw,
            "data_in_block_size": [1, x_bs],
            "data_in_skip_first_dim": x_skip,

            "bias_width": b_width,
            "bias_exponent_width": b_exp,
            "bias_exponent_bias_width": b_ebw,
            "bias_block_size": [b_bs],
        }
        return filter_kwargs_by_signature(layer_cls, kwargs)

    if layer_cls is LinearBlockLog:
        w_width = trial.suggest_categorical(f"{layer_name}_w_blog_width", SEARCH_SPACE["blog_width"])
        x_width = trial.suggest_categorical(f"{layer_name}_x_blog_width", SEARCH_SPACE["blog_width"])
        b_width = trial.suggest_categorical(f"{layer_name}_b_blog_width", SEARCH_SPACE["blog_width"])

        w_ebw = trial.suggest_categorical(f"{layer_name}_w_blog_exp_bias_width", SEARCH_SPACE["blog_exp_bias_width"])
        x_ebw = trial.suggest_categorical(f"{layer_name}_x_blog_exp_bias_width", SEARCH_SPACE["blog_exp_bias_width"])
        b_ebw = trial.suggest_categorical(f"{layer_name}_b_blog_exp_bias_width", SEARCH_SPACE["blog_exp_bias_width"])

        w_bs = trial.suggest_categorical(f"{layer_name}_w_blog_block", SEARCH_SPACE["blog_block_size"])
        x_bs = trial.suggest_categorical(f"{layer_name}_x_blog_block", SEARCH_SPACE["blog_block_size"])
        b_bs = trial.suggest_categorical(f"{layer_name}_b_blog_block", SEARCH_SPACE["blog_block_size"])

        x_skip = True

        kwargs["config"] = {
            "weight_width": w_width,
            "weight_exponent_bias_width": w_ebw,
            "weight_block_size": [w_bs],

            "data_in_width": x_width,
            "data_in_exponent_bias_width": x_ebw,
            "data_in_block_size": [1, x_bs],
            "data_in_skip_first_dim": x_skip,

            "bias_width": b_width,
            "bias_exponent_bias_width": b_ebw,
            "bias_block_size": [b_bs],
        }
        return filter_kwargs_by_signature(layer_cls, kwargs)
    if layer_cls is LinearBinary:
        w_stoch = trial.suggest_categorical(f"{layer_name}_bin_weight_stochastic",
                                            SEARCH_SPACE["bin_weight_stochastic"])
        w_bipolar = trial.suggest_categorical(f"{layer_name}_bin_weight_bipolar",
                                              SEARCH_SPACE["bin_weight_bipolar"])
        kwargs["config"] = {
            "weight_stochastic": w_stoch,
            "weight_bipolar": w_bipolar,
        }
        return filter_kwargs_by_signature(layer_cls, kwargs)
    
    if layer_cls is LinearBinaryScaling:
        cfg = {
            "data_in_stochastic": trial.suggest_categorical(
                f"{layer_name}_bins_x_stochastic", SEARCH_SPACE["bin_data_in_stochastic"]
            ),
            "bias_stochastic": trial.suggest_categorical(
                f"{layer_name}_bins_b_stochastic", SEARCH_SPACE["bin_bias_stochastic"]
            ),
            "weight_stochastic": trial.suggest_categorical(
                f"{layer_name}_bins_w_stochastic", SEARCH_SPACE["bin_weight_stochastic"]
            ),

            "data_in_bipolar": trial.suggest_categorical(
                f"{layer_name}_bins_x_bipolar", SEARCH_SPACE["bin_data_in_bipolar"]
            ),
            "bias_bipolar": trial.suggest_categorical(
                f"{layer_name}_bins_b_bipolar", SEARCH_SPACE["bin_bias_bipolar"]
            ),
            "weight_bipolar": trial.suggest_categorical(
                f"{layer_name}_bins_w_bipolar", SEARCH_SPACE["bin_weight_bipolar"]
            ),

            "binary_training": trial.suggest_categorical(
                f"{layer_name}_bins_binary_training", SEARCH_SPACE["bin_binary_training"]
            ),
        }
        kwargs["config"] = cfg
        return filter_kwargs_by_signature(layer_cls, kwargs)

    if layer_cls is LinearBinaryResidualSign:
        cfg = {
            "bypass": trial.suggest_categorical(
                f"{layer_name}_binrs_bypass", SEARCH_SPACE["binrs_bypass"]
            ),
            "data_in_levels": trial.suggest_categorical(
                f"{layer_name}_binrs_levels", SEARCH_SPACE["binrs_levels"]
            ),
            "data_in_residual_sign": trial.suggest_categorical(
                f"{layer_name}_binrs_residual_sign", SEARCH_SPACE["binrs_residual_sign"]
            ),

            "data_in_stochastic": trial.suggest_categorical(
                f"{layer_name}_binrs_x_stochastic", SEARCH_SPACE["bin_data_in_stochastic"]
            ),
            "weight_stochastic": trial.suggest_categorical(
                f"{layer_name}_binrs_w_stochastic", SEARCH_SPACE["bin_weight_stochastic"]
            ),
            "data_in_bipolar": trial.suggest_categorical(
                f"{layer_name}_binrs_x_bipolar", SEARCH_SPACE["bin_data_in_bipolar"]
            ),
            "weight_bipolar": trial.suggest_categorical(
                f"{layer_name}_binrs_w_bipolar", SEARCH_SPACE["bin_weight_bipolar"]
            ),
            "binary_training": trial.suggest_categorical(
                f"{layer_name}_binrs_binary_training", SEARCH_SPACE["bin_binary_training"]
            ),
        }
        kwargs["config"] = cfg
        return filter_kwargs_by_signature(layer_cls, kwargs)


    return filter_kwargs_by_signature(layer_cls, kwargs)


def construct_model(trial):
    trial_model = deepcopy(base_model)

    for name, layer in trial_model.named_modules():
        if not isinstance(layer, torch.nn.Linear):
            continue

        chosen_cls = trial.suggest_categorical(f"{name}_type", SEARCH_SPACE["linear_layer_choices"])

        if chosen_cls is torch.nn.Linear:
            continue

        kwargs = build_layer_kwargs(trial, name, chosen_cls, layer.in_features, layer.out_features)
        new_layer = chosen_cls(**kwargs)

        new_layer.weight.data = layer.weight.data
        if getattr(layer, "bias", None) is not None and getattr(new_layer, "bias", None) is not None:
            new_layer.bias.data = layer.bias.data

        deepsetattr(trial_model, name, new_layer)

    return trial_model

def run_mixed_precisions(n_trials: int):
    def objective(trial):
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
        return eval_results["eval_accuracy"]

    study = optuna.create_study(
        direction="maximize",
        study_name=f"bert-tiny-mixed",
        sampler=TPESampler(),
    )
    study.optimize(objective, n_trials=n_trials)

    vals = [t.value for t in study.trials if t.value is not None]
    best = []
    cur = float("-inf")
    for v in vals:
        cur = max(cur, v)
        best.append(cur)
    return best, study


if __name__ == "__main__":
    TRIALS = 50
    tokenizer_checkpoint = "bert-base-uncased"
    dataset_name = "imdb"

    with open(
        "/home/roy/Documents/STUDY/IC/YEAR4_TERM2/ADLS/mase/lab2/tutorial_5_best_model.pkl",
        "rb",
    ) as f:
        base_model = dill.load(f)

    dataset, tokenizer = get_tokenized_dataset(
        dataset=dataset_name,
        checkpoint=tokenizer_checkpoint,
        return_tokenizer=True,
    )

    curves = {}
    studies = {}

    print(f"\n==== Running mixed precision ====")
    best_curve, study = run_mixed_precisions(TRIALS)
    curves["mixed"] = best_curve
    studies["mixed"] = study
    #save best model from study
    best_trial = study.best_trial
    best_model = construct_model(best_trial)
    torch.save(best_model.state_dict(), "best_mixed_precision_model.pt")
    # print model architecture to text file
    with open("best_mixed_precision_model_architecture.txt", "w") as f:
        print(best_model, file=f)

    max_len = max(len(c) for c in curves.values() if len(c) > 0)

    plt.figure()
    for pname, c in curves.items():
        if len(c) == 0:
            continue
        if len(c) < max_len:
            c = c + [c[-1]] * (max_len - len(c))
        plt.plot(range(1, max_len + 1), c, label=pname)

    plt.xlabel("Number of trials")
    plt.ylabel("Best accuracy so far")
    plt.title("Precision Comparison: best-so-far accuracy vs trials")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("precision_comparison_best_so_far.png", dpi=200)
    plt.show()
