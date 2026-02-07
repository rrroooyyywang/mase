# Logbook

## Lab 0
### why fx?
- **High-level IR**: unlike `LLVM` or `MLIR`, `FX` offers a high-level representation of the computation which enables fast optimizations.

- **Pytorch native**: every operator in the FX graph correlates to a Python object or callable, meaning we can transform and optimize the graph, then simply regenerate the Python code required to run it. Unlike ONNX, there is no requirement for a dedicated runtime: all you need is Python.
### Bert mase graph output:
[link to the graph](../bert-base-uncased.svg)

### fx node types:
- **placeholder**: represents a function input, which can be a `Tensor` or another Python object.

- **get_attr**: retrieves a parameter from the Pytorch module hierarchy. `target` is the fully-qualified string name of the parameter’s position in the module hierarchy.

- **call_function**: applies a free function to some values. `target` is a handle to the Python callable. `args` and `kwargs` represent the arguments to the function, following the Python calling convention.

- **call_module**: applies a module in the module hierarchy’s `forward()` method with the given arguments. `target` is the fully-qualified string name of the module in the module hierarchy to call.

- **call_method**: calls a method on a value. `target` is the string name of the method to apply to the self argument.

- **output**: contains the output of the traced function in its args[0] attribute. This corresponds to the `return` statement in the Graph printout.
### `call_function` `call_module` and `call_method`
function -> any python callable functions
module -> torch.nn.module
method -> Tensor.methods

```
random_tensor = torch.randn(2, 2)

function_relu = torch.relu(random_tensor)
method_relu = random_tensor.relu()
module_relu = torch.nn.ReLU()(random_tensor)
```

### Task: Delete the call to `replace_all_uses_with` to verify that FX will report a RuntimeError.
**Result:**
![](./img/lab0_task1.png)

### Task: Remove the `attention_mask` and `labels` arguments from the `hf_input_names` list and re-run the following cell. Use `mg.draw()` to visualize the graph in each case. Can you see any changes in the graph topology? Can you explain why this happens?

[click for full figure: 1 inputs](./img/lab0_tutorial_2_task_1_removed.svg)
[click for full figure: 3 inputs](./img/lab0_tutorial_2_task_1_full.svg)

**Compering:**
![](./img/laba0_tutorial_2_task_1_compare_1.png)
The aboved figure shows the 3 inputs graph will include a crossentropy module which allow the modole to output the loss by passing the `labels`.

![](./img/laba0_tutorial_2_task_1_compare_2.png)
The aboved figure shows the 3 inputs graph will include a input for the attention mask which will be usfull when **training in a batch**.

Therefore, I know that 3 input is required for traning, but 1 input is for inference.
## Lab 1

### Task 1:
In Tutorial 3, you quantized every Linear layer in the model to the provided configuration. Now, explore a range of fixed point widths from 4 to 32.

- a: Plot a figure where the x-axis is the fixed point width and the y-axis is the highest achieved accuracy on the IMDb dataset, following the procedure in Tutorial 3.

![](./img/lab1_tutorial_3_task_1_a.png)

- b. Plot separate curves for PTQ and QAT at each precision to show the effect of post-quantization finetuning.

![](./img/lab1_tutorial_3_task_1_b.png)

```python
import matplotlib.pyplot as plt
import chop.passes as passes

def quant_train_sweep(int_widths=range(4, 32),frac_rule=lambda w: w // 2,qat_epochs=1):

    ptq_accs = []
    qat_accs = []

    for int_width in int_widths:
        int_frac = frac_rule(int_width)
        print(f"\n=== width={int_width}, frac={int_frac} ===")

        # PTQ
        mg = MaseGraph.from_checkpoint(
            "/home/roy/Documents/STUDY/IC/YEAR4_TERM2/ADLS/mase/lab0/tutorial_2_lora"
        )
        trainer_ptq = get_trainer(
            model=mg.model,
            tokenized_dataset=dataset,
            tokenizer=tokenizer,
            evaluate_metric="accuracy",
        )

        print(f"[PTQ] Quantizing (width={int_width}, frac={int_frac}) then evaluate...")
        quantization_config_ptq = quant_config(int_width=int_width, int_frac_width=int_frac)
        mg_ptq, _ = passes.quantize_transform_pass(mg, pass_args=quantization_config_ptq)

        trainer_ptq.model = mg_ptq.model
        ptq_eval = trainer_ptq.evaluate()
        ptq_acc = ptq_eval["eval_accuracy"]
        print(f"[PTQ] eval_accuracy = {ptq_acc:.6f}")
        ptq_accs.append(ptq_acc)

        # QAT
        mg2 = MaseGraph.from_checkpoint(
            "/home/roy/Documents/STUDY/IC/YEAR4_TERM2/ADLS/mase/lab0/tutorial_2_lora"
        )
        trainer_qat = get_trainer(
            model=mg2.model,
            tokenized_dataset=dataset,
            tokenizer=tokenizer,
            evaluate_metric="accuracy",
        )

        print(f"[QAT] Quantizing then finetune for {qat_epochs} epoch(s)...")
        quantization_config_qat = quant_config(int_width=int_width, int_frac_width=int_frac)
        mg_qat, _ = passes.quantize_transform_pass(mg2, pass_args=quantization_config_qat)
        
        trainer_qat.model = mg_qat.model
        device = trainer_qat.args.device
        trainer_qat.model.to(device)
        trainer_qat.model.train()
        trainer_qat.train()


        qat_eval = trainer_qat.evaluate()
        best_acc = qat_eval["eval_accuracy"]

        print(f"[QAT] best achieved accuracy = {best_acc:.6f}")
        qat_accs.append(best_acc)

    plt.figure(figsize=(10, 6))
    plt.plot(list(int_widths), ptq_accs, marker="o", label="PTQ (eval after quant)")
    plt.plot(list(int_widths), qat_accs, marker="o", label=f"QAT (finetune {qat_epochs} ep, best achieved)")
    plt.title("PTQ vs QAT: Accuracy vs Fixed-Point Width")
    plt.xlabel("Fixed-Point Width (bits)")
    plt.ylabel("Accuracy")
    plt.xticks(list(int_widths))
    plt.grid(True)
    plt.legend()
    plt.show()

    return ptq_accs, qat_accs

```

### Task 2:
Take your best obtained model from Task 1 and rerun the pruning procedure, this time varying the sparsity from 0.1 to 0.9.

- a: Plot a figure where the x-axis is the sparsity and the y-axis is the highest achieved accuracy on the IMDb dataset, following the procedure in Tutorial 4.
![](./img/lab1_tutorial_4_task_2_a.png)
- b: Plot separate curves for Random and L1-Norm methods to evaluate the effect of different pruning strategies.
![](./img/lab1_tutorial_4_task_2_b_2.png)

```python
import matplotlib.pyplot as plt
import torch
import chop.tools as tools
def pruning_config(sparsity, method="l1-norm", scope="local"):
    return {
        "weight": {
            "sparsity": sparsity,
            "method": method,
            "scope": scope,
        },
        "activation": {
            "sparsity": sparsity,
            "method": method,
            "scope": scope,
        },
    }

def get_best_accuracy(trainer):
    best_acc = None

    trainer.train()

    if hasattr(trainer, "state") and hasattr(trainer.state, "log_history"):
        for rec in trainer.state.log_history:
            if "eval_accuracy" in rec:
                best_acc = (
                    rec["eval_accuracy"]
                    if best_acc is None
                    else max(best_acc, rec["eval_accuracy"])
                )

    if best_acc is None:
        best_acc = trainer.evaluate()["accuracy"]

    return best_acc


def sparsity_sweep(batch_size=8, log_path="pruning_results.txt"):
    sparsity_list = [i / 10 for i in range(1, 9)]
    random_accuracies = []
    l1_accuracies = []

    with open(log_path, "a") as f:

        print("Random pruning sweep...")
        for sparsity in sparsity_list:
            print(f"[Random] sparsity={sparsity}")

            mg = MaseGraph.from_checkpoint(
                "/home/roy/Documents/STUDY/IC/YEAR4_TERM2/ADLS/mase/lab1/tutorial_3_qat"
            )
            mg, _ = passes.prune_transform_pass(
                mg, pass_args=pruning_config(sparsity, method="random")
            )

            trainer = tools.get_trainer(
                model=mg.model,
                tokenized_dataset=dataset,
                tokenizer=tokenizer,
                evaluate_metric="accuracy",
                num_train_epochs=5,
                train_batch_size=batch_size,
            )

            device = torch.device("cuda:0")
            trainer.model.to(device)
            trainer.train()

            acc = trainer.evaluate()["eval_accuracy"]
            random_accuracies.append(acc)

            f.write(f"random {sparsity:.1f} {acc:.6f}\n")
            f.flush()

        print("L1-Norm pruning sweep...")
        for sparsity in sparsity_list:
            print(f"[L1] sparsity={sparsity}")

            mg = MaseGraph.from_checkpoint(
                "/home/roy/Documents/STUDY/IC/YEAR4_TERM2/ADLS/mase/lab1/tutorial_3_qat"
            )
            mg, _ = passes.prune_transform_pass(
                mg, pass_args=pruning_config(sparsity, method="l1-norm")
            )

            trainer = tools.get_trainer(
                model=mg.model,
                tokenized_dataset=dataset,
                tokenizer=tokenizer,
                evaluate_metric="accuracy",
                num_train_epochs=5,
                train_batch_size=batch_size,
            )

            device = torch.device("cuda:0")
            trainer.model.to(device)
            trainer.train()

            acc = trainer.evaluate()["eval_accuracy"]
            l1_accuracies.append(acc)

            f.write(f"l1-norm {sparsity:.1f} {acc:.6f}\n")
            f.flush()

    plt.figure(figsize=(10, 6))
    plt.plot(sparsity_list, random_accuracies, marker="o", label="Random pruning")
    plt.plot(sparsity_list, l1_accuracies, marker="o", label="L1-Norm pruning")
    plt.xlabel("Sparsity")
    plt.ylabel("Accuracy")
    plt.title("Pruning Sparsity vs Accuracy (IMDb)")
    plt.grid(True)
    plt.legend()
    plt.show()

    return sparsity_list, random_accuracies, l1_accuracies

```
## Lab 2
### Task 1
Tutorial 5 shows how to use random search to find the optimal configuration of hyperparameters and layer choices for the Bert model.
- a: Now, explore using the GridSampler and TPESampler in Optuna.
In file [nas.py](../lab2/nas.py)
- b: Plot a figure that has the number of trials on the x axis, and the maximum achieved accuracy up to that point on the y axis. Plot one curve for each sampler to compare their performance.
![](./img/lab2_task1_b.png)
### Task 2
In Tutorial 5, NAS is used to find an optimal configuration of hyperparameters, then we use the CompressionPipeline in Mase to quantize and prune the model after search is finished. However, the final compressed model may not be optimal, since different model architectures may have different sensitivities to quantization and pruning. Ideally, we want to run a compression-aware search flow, where the quantization and pruning is considered in each trial.

- a: In the objective function, after the model is constructed and trained for some iterations, call the CompressionPipeline to quantize and prune the model, then continue training for a few more epochs. Use the sampler that yielded the best results in Task 1 to run the compression-aware search. The objective function should return the final accuracy of the model after compression. Consider also the case where final training is performed after quantization/pruning.

- b: Plot a new figure that has the number of trials on the x axis, and the maximum achieved accuracy up to that point on the y axis. There should be three curves: 1. the best performance from Task 1 (without compression), compression-aware search without post-compression training, and compression-aware search with post-compression training.
![](./img/lab2_task2_b.png)
## Lab 3
### Task 1:
In Tutorial 6, all layers allocated to IntegerLinear are allocated the same width and fractional width. This is suboptimal, as different layers may have different sensitivities to quantization.

- a: Modify the code to allow different layers to have widths in the range [8, 16, 32] and fractional widths in the range [2, 4, 8]. Expose this choice as an additional hyperparameter for the Optuna sampler.
In file [mps.py](../lab3/mps.py)
- b: Run the search again, and plot a figure that has the number of trials on the x axis, and the maximum achieved accuracy up to that point on the y axis.
![](./img/lab3_task1.png)

### Task 2:
![](./img/lab3_task2.png)

## Lab 4

### Task 1:
- case: CPU

Original model: `0.9344 s`

Optimized model: `2.5768 s`

much slower! The possible reasons:

1) ResNet-18 is dominated by convolution layers, which are already highly optimized in PyTorch eager mode using oneDNN on CPU. TorchInductor primarily benefits models with heavy pointwise operations or fusion opportunities. For convolution-heavy models, the generated code often cannot outperform optimized vendor libraries.

2) My cpu is Intel i7-12700KF, which has multi cpus. Different execution paths may use different threading strategies. TorchInductor and eager mode can interact differently with OpenMP or oneDNN thread pools, sometimes leading to oversubscription or inefficient core utilization, which can degrade performance.

3) torch.compile is jit compilation. This intruduced compilation overhead for each bytecode executaion.

- case: GPU

GPU Original model: `0.0411 s`

GPU Optimized model: `0.3081 s`

even much more slower! The possible reasons:
1) During compiling trail, I got some warning as output, this may be due to some overhead happed when the compile run the model.forward in first time, for example. looking for correct liberary and the low level apis (pynvml). And also during the backend searching or loading, the compile decide to used some bad backends.

2) Maybe there are other threads are using the GPU and the GPU is too busy to run the trail.

- case: after warmup
After the two initila trail, I ran it again in notebook, the result is much better.

CPU Original model: `0.9321 s`

CPU Optimized model: `0.7850 s`

GPU Original model: `0.0214 s`

GPU Optimized model: `0.0213 s`


1) The possible case is that the model is being cache inside the device and the comiled path is also cached, no more jit graph break and recompiling happend in the runtime.
