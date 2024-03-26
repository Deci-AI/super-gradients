# Parameters check and Unused parameters

Super-Gradients features string based parameters, which means that you can define the parameters for 
the model, the dataset, the training or any other part of the system by passing structured dictionaries 
of strings (from YAML files or directly from the code).

This approach allows high level of flexibility and can support fast and simple changes to the code, but it also poses a potential risks - a small mistake or a typo 
in a parameter name may cause your training to take an unexpected direction.

To address this issue, Super-Gradients includes two features:
1. Fuzzy matching - if your parameter name has only a small typo, it will still be matched with a close parameter name. 
I.e. `vel_dataloader` parameter will be identified as `val_dataloader`.
2. Unused parameters check - Super-Gradients expects all the parameters in the dictionary to be consumed by the code. This feature
was meant to avoid a situation where the user passes a parameter but that parameter is not defined anywhere in the code and has no affect.


## Unused Parameters Detection
Super-Gradients checks for unused parameters in:
 - Training hyperparameters
 - Dataset (and Dataloader) parameters
 - Model architecture parameters

When this check fails, you will see the following error message, and the execution will terminate:
```commandline
super_gradients.training.utils.config_utils.UnusedConfigParamException: Detected unused parameters in configuration object that were not consumed by caller: 
	number_of_images
```
In the example above, "number_of_images" is the unused parameter name.

In the code itself, this check will look as follows:
```python
from super_gradients.training.utils import warn_if_unused_params
    
with check_if_unused_params(some_config) as some_config_wrapped:
    do_something_with_config(some_config_wrapped)
```
All of the parameters in the `some_config_wrapped` are expected to be consumed within the code of `do_something_with_config`.
The wrapped config can be a `list`, `tuple`, `ListConfig`, `dict`, `Mapping`, `DictConfig`, `HpmStruct`, or `TrainingParams` object.

Note that any request for a parameter will consume the parameter. I.e. the following example will pass without any error:
```python

params = {"a": 1, 
          "b": ["x", "y"],
          "c": {
              "c1": "hello",
              "c2": False,
          }}

def sum(text: str, _print: bool):
    if _print
        print(text)
    
with check_if_unused_params(params) as params_wrapped:
    if params_wrapped["a"] == 1:
        for letter in params_wrapped["b"]:
            print(letter)
    
    _pass_parameters_by_name(sum, params["c"])
```
**[!] Important:** In this example, we used the `_pass_parameters_by_name` function. Usually, we could have just called `sum(**params["c"])` but the `**` operator does not expose
the parameters retrival. Instead of `**` use the `_pass_parameters_by_name` function.

Two more function that can be used instead of `check_if_unused_params` are `raise_if_unused_params` and `warn_if_unused_params` functions.


### Change behavior:
You can avoid the execution termination by setting the `UNUSED_PARAMS_ACTION` environment variable.
```commandline
export UNUSED_PARAMS_ACTION=warn
```
The three valid values are: `warn`, `raise`, and `ignore`

### Consume the parameter early
If your parameter is consumed later in the code (i.e. after a few epochs), save it locally - this will consume the parameter

```python

params = {"a": 1, 
          "b": ["x", "y"],
          }
    
with check_if_unused_params(params) as params_wrapped:
    self.a = params_wrapped["a"]
...
do something
...
print(self.a)

```
