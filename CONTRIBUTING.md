# Contribution Guidelines

We appreciate all contributions. If you are planning to contribute back bug-fixes, please do so without any further discussion. If you plan to contribute new features, utility functions or extensions, please first open an issue and discuss the feature with us.

Here are a few more things to know

- [How to Contirbute](#how-to-contribute)
- [Jupyter Notebooks Contribution](#jupyter-notebooks-contribution)
- [Code Style Guidelines](#code-style-guidelines)
    
## How to Contribute

Here is a simple guideline to get you started with your first contribution
1. Use [issues](https://github.com/Deci-AI/super-gradients/issues) to discuss the suggested changes. Create an issue describing changes if necessary and add labels to ease orientation.
1. [Fork super-gradients](https://help.github.com/articles/fork-a-repo/) so you can make local changes and test them.
1. Create a new branch for the issue. The branch naming convention is enforced by the CI/CD so please make sure you are using your_username/your_branch_name convention otherwise it will fail.
1. Create relevant tests for the issue, please make sure you are covering unit, integration and e2e tests where required.
1. Make code changes.
1. Ensure all the tests pass and code formatting is up to standards, and follows [PEP8](https://www.python.org/dev/peps/pep-0008/).
<!--1. We use [pre-commit](https://pre-commit.com/) package to run our pre-commit hooks. Black formatter and flake8 linter will be ran on each commit. In order to set up pre-commit on your machine, follow the steps here, please note that you only need to run these steps the first time you use pre-commit for this project.)

    * Install pre-commit from pypi
   ```
    $ pip install pre-commit
   ```    
    * Set up pre-commit inside super-gradients repo which will create .git/hooks directory.
   ```
   $ pre-commit install
   ```
   ```
   $ git commit -m "message"
   ```
   
    * Each time you commit, git will run the pre-commit hooks (black and flake8 for now) on any python files that are getting committed and are part of the git index.  If black modifies/formats the file, or if flake8 finds any linting errors, the commit will not succeed. You will need to stage the file again if black changed the file, or fix the issues identified by flake8 and and stage it again.

    * To run pre-commit on all files just run
   ```
   $ pre-commit run --all-files
   ```
-->
7. Create a pull request against <b>master</b> branch.



## Jupyter Notebooks Contribution

Pulling updates from remote might cause merge conflicts with jupyter notebooks. The tool [nbdime](https://nbdime.readthedocs.io/en/latest/) might solve this.
* Installing nbdime
```
pip install ndime
```
* Run a diff between two notebooks
```
nbdiff notebook_1.ipynb notebook_2.ipynb
```

## Code Style Guidelines

We are working hard to make sure all the code in this repository is readable, maintainable and testable.
We follow the Google docstring guidelines outlined on this [styleguide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings) page. For example:
```python
  def python_function(first_argument: int, second_argument: int) -> bool:
      """
      This function will do something with the two arguments.

      Args:
          first_argument (int): the first argument to the function
          second_argument (int): the second argument to the function

      Raises:
          Exception: arguments should be ints not floats

      Returns:
          bool: whether or not the calculation was correct
      """
```


## Documentation

We use  GitHub Pages for technical documentation hosting on https://docs.deci.ai/super-gradients/documentation/source/welcome.html <br>
To generate the docs based on the current work tree, run: <br>
<code>./scripts/generate_docs.sh</code> <br><br>
And the documentation will automatically update, based on <code>documentation/</code>. <br>
The new documentation HTML will be generated to <code>docs/</code>. <br> 
Once <code>docs/</code> is committed and pushed, GitHub Pages will use it.<br>
The step of documentation update is currently manual.
