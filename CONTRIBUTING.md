# Contribution Guidelines

Here is a simple guideline to get you started with your first contribution.
1. Set up your environment to follow our [formatting guidelines](#code-formatting) and to use [signed-commits](#signed-commits).
2. Use [issues](https://github.com/Deci-AI/super-gradients/issues) to discuss the suggested changes. Create an issue describing changes if necessary and add labels to ease orientation.
3. [Fork super-gradients](https://help.github.com/articles/fork-a-repo/) so you can make local changes and test them.
4. Create a new branch for the issue. The branch naming convention is enforced by the CI/CD so please make sure you are using `feature/SG-***` or `hotfix/SG-***` format otherwise it will fail.
5. Implement your changes along with relevant tests for the issue. Please make sure you are covering unit, integration and e2e tests where required.
6. Create a pull request against **master** branch.



## Code Style

We follow the **reStructuredText** docstring format (default of PyCharm), along with typing.
```python
def python_function(first_argument: int, second_argument: int) -> str:
    """Do something with the two arguments.

    :param first_argument: First argument to the function
    :param second_argument: Second argument to the function
    :return: Description of the output
    """
```




## Code Formatting

We enforce [black](https://github.com/psf/black) code formatting in addition to existing [flake8](https://flake8.pycqa.org/en/latest/user/index.html) checks. 

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
For flake8: ```$ flake8 --statistics --config scripts/flake8-config setup.py .``` 


## Signed Commits

### Background

Signed commits provide a way to verify the authenticity and integrity of the code changes made by a particular developer, as the commit is cryptographically signed using their private GPG key. This helps ensure that the code changes were made by the intended person and have not been tampered with during transit.

You can find more information [here](https://withblue.ink/2020/05/17/how-and-why-to-sign-git-commits.html).

### Add GPG key to GitHub

1. [Generate a new GPG key](https://docs.github.com/en/authentication/managing-commit-signature-verification/generating-a-new-gpg-key). After completing the steps 1-9 you'll see the console message containing generated key ID (e.g. <i>3AA5C34371567BD2</i>).
2. Copy the GPG key by running the command on step 12 from the link above
    
```bash
$ gpg --armor --export <YOUR_GPG_KEY_ID>
```
    
3. [Add the new GPG key to your GitHub account](https://docs.github.com/en/authentication/managing-commit-signature-verification/adding-a-new-gpg-key-to-your-github-account)

### Use GPG key

- [From Pycharm](https://www.jetbrains.com/help/pycharm/set-up-GPG-commit-signing.html#enable-commit-signing)
- [From Terminal](https://docs.github.com/en/authentication/managing-commit-signature-verification/signing-commits), but first also do:
    
```bash
$ git config --global user.signingkey <YOUR_GPG_KEY_ID>
$ git config --global gpg.program $(which gpg)
$ git config --global commit.gpgsign true
```

### Making signed commits
Use ```-S``` flag to sign your commit:
```bash
git commit -S -m "Making my first signed commit"
```


You could run the next command to ensure the commit has been signed:
```bash
git log --show-signature -1
```

### I have created PR with unsigned commits. Do I have to start over?

No. What you need to do is create another branch starting from master, then move your commits from PR branch to the 
new branch and force push it to the remote under the old name. Let's say your PR branch named 'feature/my_awesome_pr'
then you need to do the following:

```bash
git checkout master
git checkout -b feature/my_awesome_pr_signed
git merge --squash --no-commit feature/my_awesome_pr
git commit -S -m "My signed commit"
git push -f origin feature/my_awesome_pr_signed:feature/my_awesome_pr
```

The last command will overwrite your PR branch with the new signed commit containing all changes from the PR. 


### GPG debug
Some of contributors can face this problem while making their first commit:
```
error: gpg failed to sign the data
fatal: failed to write commit object
```
You could consider adding ```GIT_TRACE=1``` at the beginning of the ```git commit``` command, this will show you what the commit command does under the hood. In case you are seeing something like this 
```bash
18:57:12.099725 run-command.c:663       trace: run_command: /usr/bin/gpg --status-fd=2 -bsau <YOUR_GPG_KEY_ID>
```
extract the GPG command (```/usr/bin/gpg --status-fd=2 -bsau <YOUR_GPG_KEY_ID>```) and execute it separately. Now you can check what happened while running the GPG signature.



### Jupyter Notebooks Contribution

Pulling updates from remote might cause merge conflicts with jupyter notebooks. The tool [nbdime](https://nbdime.readthedocs.io/en/latest/) might solve this.
* Installing nbdime
```
pip install nbdime
```
* Run a diff between two notebooks
```
nbdiff notebook_1.ipynb notebook_2.ipynb
```

