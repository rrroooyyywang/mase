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
## Lab 1

## Lab 2

## Lab 3