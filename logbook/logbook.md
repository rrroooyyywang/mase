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

## Task 1:
In Tutorial 3, you quantized every Linear layer in the model to the provided configuration. Now, explore a range of fixed point widths from 4 to 32.

- a: Plot a figure where the x-axis is the fixed point width and the y-axis is the highest achieved accuracy on the IMDb dataset, following the procedure in Tutorial 3.

![](./img/lab1_tutorial_3_task_1_a.png)

- b. Plot separate curves for PTQ and QAT at each precision to show the effect of post-quantization finetuning.

![](./img/lab1_tutorial_3_task_1_b.png)
## Lab 2

## Lab 3