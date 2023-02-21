# Contribution Guidelines

Here is a simple guideline to get you started with your first contribution
2. Set up your environment to follow our [formatting guidelines](#code-formatting) and to use [signed-commits](#signed-commits).
3. Use [issues](https://github.com/Deci-AI/super-gradients/issues) to discuss the suggested changes. Create an issue describing changes if necessary and add labels to ease orientation.
4. [Fork super-gradients](https://help.github.com/articles/fork-a-repo/) so you can make local changes and test them.
5. Create a new branch for the issue. The branch naming convention is enforced by the CI/CD so please make sure you are using `feature/SG-***` or `hotfix/SG-***` format otherwise it will fail.
6. Implement your changes along with relevant tests for the issue. Please make sure you are covering unit, integration and e2e tests where required.
7. Create a pull request against <b>master</b> branch.



## Code Style

We are working hard to make sure all the code in this repository is readable, maintainable and testable.
```python
def python_function(first_argument: int, second_argument: int) -> str:
    """Do something with the two arguments.

    :param first_argument: First argument to the function
    :param second_argument: Second argument to the function
    :return: Description of the output
    """
```




## Code Formatting

We enforce [black](https://github.com/psf/black) code formatting in addition to existing flake8 checks. 

To ensure everyone uses same code style, a project-wise [configuration file](https://github.com/Deci-AI/super-gradients/blob/master/pyproject.toml) has been added to SG repo. It ensures all formatting will be exactly the same regardless of OS, python version or the place where code formatting check is happening. 

### Installation

To start, one need to install required development dependencies (actual versions of black, flake8 and git commit hooks):

`$ pip install -r requirements.dev.txt`

### Pre-Commit Hooks

A pre-commit hook as an easy way to ensure all files in the commit are already formatted accordingly and pass linter checks. If they are not, the git will prevent commit of problematic files unless errors are fixed.

To start, run the following command from SG repo root:
```bash
$ pip install pre-commit
$ pre-commit install
```

The command should complete without errors. Once done, all your upcoming commits will be checked via black & flake8. 

### Usage

Just run ```$ black .``` from the SG root. It will reformat the whole repo.




## Signed Commits

### Background

Signed commits provide a way to verify the authenticity and integrity of the code changes made by a particular developer, as the commit is cryptographically signed using their private GPG key. This helps ensure that the code changes were made by the intended person and have not been tampered with during transit.

You can find more information [here](https://withblue.ink/2020/05/17/how-and-why-to-sign-git-commits.html).

### Add GPG key to GitHub

1. [Generate a new GPG key](https://docs.github.com/en/authentication/managing-commit-signature-verification/generating-a-new-gpg-key) 
2. Copy the GPG key by running the command on step 12 from the link above
    
    ```bash
    $ gpg --armor --export 3AA5C34371567BD2
    ```
    
3. [Add the new GPG key to your GitHub account](https://docs.github.com/en/authentication/managing-commit-signature-verification/adding-a-new-gpg-key-to-your-github-account)

### Use GPG key

- [From Pycharm](https://www.jetbrains.com/help/pycharm/set-up-GPG-commit-signing.html#enable-commit-signing)
- [From Terminal](https://docs.github.com/en/authentication/managing-commit-signature-verification/signing-commits), but first also do:
    
    ```bash
    $ git config --global user.signingkey 3AA5C34371567BD2
    $ git config --global gpg.program /usr/local/bin/gpg
    $ git config --global commit.gpgsign true
    ```


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
