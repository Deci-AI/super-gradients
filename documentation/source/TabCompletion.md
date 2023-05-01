# Tab Completion

SuperGradients configuration system uses Hydra, so it is possible to enable [tab completion](https://hydra.cc/docs/tutorials/basic/running_your_app/tab_completion/) feature.

## Installation

To enable tab completion you need to run following commands:

Example for train_from_recipe.py (Command to be executed from `src/super_gradients`)

`eval "$(python train_from_recipe.py -sc install=bash)"`

If you're using ZSH (Mac OS X), then ensure that you have the following lines in your `.zshrc`:
```
autoload -Uz compinit
compinit
autoload -Uz bashcompinit && bashcompinit
```

See [Tab Completion](https://hydra.cc/docs/tutorials/basic/running_your_app/tab_completion/) for more information.

## Caveats & Limitations

* Tab completion is not supported on Windows.
* You need to enable tab completion explicitly for each entry point in your application. In other words, if your project.
contains two train scripts (train_a.py and train_b.py), you need to enable tab completion for each entry point separately:
```bash
eval "$(python train_a.py -sc install=bash)"
eval "$(python train_b.py -sc install=bash)"
```
* Tab completion is not supported for running train_from_recipe using `-m` form: `python -m super_gradients.train_from_recipe ...`.
* If you don't have tab completion in zsh shell, ensure that your `.zshrc` file contains following lines:
```
autoload -Uz compinit
compinit
autoload -Uz bashcompinit && bashcompinit
```

If it doesn't - add lines above to your `.zshrc` file and restart the shell.
* Tab completion may become unusably slow for scripts that contains many top-level imports in it. 
It is recommended to reduce number of top-level imports to the absolute minimum (hydra + OmegaConf) and import the 
remaining dependencies inside the main function.
