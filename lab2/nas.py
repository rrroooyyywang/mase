import argparse
from pyexpat import model
from chop.tools import get_tokenized_dataset
import torch.nn as nn
from chop.nn.modules import Identity
from transformers import AutoConfig, AutoModelForSequenceClassification
from chop.tools.utils import deepsetattr
from chop.tools import get_trainer
from optuna.samplers import GridSampler, RandomSampler, TPESampler
import optuna
from optuna.trial import TrialState
import math

def construct_model(trial, do_layer_search=True):
    config = AutoConfig.from_pretrained(checkpoint)

    # Update the paramaters in the config
    for param in [
        "num_layers",
        "num_heads",
        "hidden_size",
        "intermediate_size",
    ]:
        chosen_idx = trial.suggest_int(param, 0, len(search_space[param]) - 1)
        setattr(config, param, search_space[param][chosen_idx])

    trial_model = AutoModelForSequenceClassification.from_config(config)

    if not do_layer_search:
        return trial_model

    for name, layer in trial_model.named_modules():
        if isinstance(layer, nn.Linear) and layer.in_features == layer.out_features:
            new_layer_cls = trial.suggest_categorical(
                f"{name}_type",
                search_space["linear_layer_choices"],
            )

            if new_layer_cls == nn.Linear:
                continue
            elif new_layer_cls == Identity:
                new_layer = Identity()
                deepsetattr(trial_model, name, new_layer)
            else:
                raise ValueError(f"Unknown layer type: {new_layer_cls}")

    return trial_model


def objective(trial):

    # Define the model
    model = construct_model(trial, SAMPLER != "grid")

    trainer = get_trainer(
        model=model,
        tokenized_dataset=dataset,
        tokenizer=tokenizer,
        evaluate_metric="accuracy",
        num_train_epochs=1,
    )

    trainer.train()
    eval_results = trainer.evaluate()

    # Set the model as an attribute so we can fetch it later
    trial.set_user_attr("model", model)

    return eval_results["eval_accuracy"]

def build_sampler(name: str):
    name = name.lower()

    if name == "random":
        print("[INFO] Using RandomSampler")
        return RandomSampler()

    elif name == "tpe":
        print("[INFO] Using TPESampler")
        return TPESampler()

    elif name == "grid":
        print("[INFO] Using GridSampler")
        grid_space = {
            "num_layers": list(range(len(search_space["num_layers"]))),
            "num_heads": list(range(len(search_space["num_heads"]))),
            "hidden_size": list(range(len(search_space["hidden_size"]))),
            "intermediate_size": list(range(len(search_space["intermediate_size"]))),
        }
        return GridSampler(grid_space)

    else:
        raise ValueError(f"Unknown sampler: {name}")

def export_trials_to_txt(study, filepath):
    trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    trials.sort(key=lambda t: t.number)

    best = -math.inf if study.direction.name == "MAXIMIZE" else math.inf

    with open(filepath, "w") as f:
        f.write("# trial_number trial_accuracy best_so_far_accuracy\n")

        for t in trials:
            acc = t.value
            if acc is None:
                continue

            if study.direction.name == "MAXIMIZE":
                best = max(best, acc)
            else:
                best = min(best, acc)

            f.write(f"{t.number} {acc:.6f} {best:.6f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sampler",
        type=str,
        default="random",
        choices=["random", "tpe", "grid"],
        help="Optuna sampler type",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=10,
        help="Number of trials",
    )
    parser.add_argument(
        "--study_name",
        type=str,
        default="bert-tiny-nas-study",
    )
    args = parser.parse_args()


    checkpoint = "prajjwal1/bert-tiny"
    tokenizer_checkpoint = "bert-base-uncased"
    dataset_name = "imdb"

    dataset, tokenizer = get_tokenized_dataset(
        dataset=dataset_name,
        checkpoint=tokenizer_checkpoint,
        return_tokenizer=True,
    )
    search_space = {
        "num_layers": [8,4,2],
        "num_heads": [16, 8, 4, 2],
        "hidden_size": [512, 384, 256, 192, 128],
        "intermediate_size": [2048, 1536, 1024, 768, 512],
        "linear_layer_choices": [
            nn.Linear,
            Identity,
        ],
    }

    SAMPLER = args.sampler.lower()
    sampler = build_sampler(args.sampler)

    study = optuna.create_study(
        direction="maximize",
        study_name=f"{args.study_name}-{args.sampler}",
        sampler=sampler,
    )

    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=60 * 60 * 24,
        show_progress_bar=True,
    )

    output_file = f"/home/roy/Documents/STUDY/IC/YEAR4_TERM2/ADLS/mase/lab2/results_{args.sampler}.txt"
    export_trials_to_txt(study, output_file)

    print(f"[INFO] Results written to {output_file}")