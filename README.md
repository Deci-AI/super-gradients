# SuperGradients

## Introduction

This repository contains a utility python package that helps individuals to train their code using SuperGradient's code.  
There are two ways you can install it on your local machine - using this GitHub repository or using SuperGradient's private PyPi
repository.

<!-- toc -->

- [Installation Methods](#installation-methods)
    - [Quick Installation of stable version](#quick-installation-of-stable-version)
    - [Installing from GitHub](#installing-from-github)
    - [Installing from AWS Codeartifact PyPi repository](#installing-from-aws-codeartifact-pypi-repository)
- [Development Flow](#development-flow)
    - [Feature and bugfix branches](#feature-and-bugfix-branches)
    - [Merging to Master](#merging-to-master)
    - [Creating a release](#creating-a-release)
- [Technical Debt](#technical-debt)

<!-- tocstop -->

## Installation Methods

### Quick Installation of stable version

While being in the context of your environment, be it `venv` or `conda`, run:

```bash
  pip install git+https://github.com/Deci-AI/deci_trainer.git@stable
or
  pip install git+ssh://git@github.com/Deci-AI/super_gradients.git@stable 
```

That's it !

### Installing from GitHub

#### Prerequisites:

1. Read access to this repository. If you can read this document, you probably have it.
2. Know the version that you want to install. This can be done by viewing
   the [Releases page](https://github.com/Deci-AI/deci_trainer/releases). It's recommended to take
   the [latest one](https://github.com/Deci-AI/deci_trainer/releases/latest)

#### Installation

Let's assume the release that you would like to install is `0.0.1`. While being in the context of your environment, be
it `venv` or `conda`, run:

```bash
  pip install git+https://github.com/Deci-AI/deci_trainer.git@0.0.1
```

That's it!

Notice that the command above is using http connection. You can alternatively use SSH by running:

```bash
  pip install git+ssh://git@github.com/Deci-AI/super_gradients.git@feature/DLE-123_my_cool_feature
```

### Installing from AWS Codeartifact PyPi repository

In order to install from Codeartifact we will connect to the remote repository on AWS and modify our Pip config file.  
As we have separate repositpries for development and production, the command changes accordingly.

#### Prerequisites:

1. Make sure that you have [access to AWS](https://github.com/Deci-AI/deci-devops/blob/master/README.md#from-the-cli)
   using `aws sts get-caller-identity`.

#### 1a. Install a Release Candidate:

run:

```bash
  aws codeartifact login --tool pip --repository deci-packages --domain deci-packages --domain-owner 307629990626 --profile deci-dev
```

#### 1b. Install a Release Candidate (mutually exclusive with 1a):

run:

```bash
  aws codeartifact login --tool pip --repository deci-packages --domain deci-packages --domain-owner 487290820248 --profile deci-prod
```

#### 2. Edit `pip.conf` config file

AWS CLI configured the access token to our private PyPi repository in your ` ~/.config/pip/pip.conf` file.  
If you will open it you should see something like this:

```bash
[global]
index-url = https://aws:eyJ2ZXIiOjEsImlzdSI6MTYxNzcwMjc.....5OSwiZW5jIjoiQTcbp1rFfe_Ir_ATZUg@deci-packages-307629990626.d.codeartifact.us-east-1.amazonaws.com/pypi/deci-packages/simple/
```

We must add `extra-` prefix to `index-url` so it will become `extra-index-url = https://...`.  
You can do so by manually edit the file with your faivorite text editor, or run the coomand:
(you must have `sed` installed)

```bash

  sed -i 's/^index-url/extra-index-url/g'  ~/.config/pip/pip.conf
```

## Development Flow

### Feature and bugfix branches

When working on a branch, you will probably want to be able to test your work locally. In order to do so while not
adding noise to our PyPi repository, you can install the package directly from GitHub. There are 2 ways doing so - same
as there are for cloning - via HTTPS and via SSH.

Assuming your branch name is `feature/DLE-123_my_cool_feature` you can either:

```bash
    pip install git+https://github.com/Deci-AI/deci_trainer.git@feature/DLE-123_my_cool_feature
```

or using ssh -

```bash
     pip install git+ssh://git@github.com/Deci-AI/super_gradients.git@feature/DLE-123_my_cool_feature
```

### Debugging flow

In order to apply new changes in the code to your local machine:

1. Push the changes into the remote repository i.e. ```git push origin feature/DLE-123_my_cool_feature```
2. ``` pip uninstall deci-trainer```
3. ```pip install git+https://github.com/Deci-AI/deci_trainer.git@feature/DLE-123_my_cool_feature ```

### Merging to Master

When you are happy with your change, create a PR to master. After a code review by one of your peers, the branch will be
merged into master. That merge will trigger an automation process that will, if successful, push a release candidate
version of the package into SuperGradient's AWS Codeartifact repository. The package will be named X.Y.Zrc${CIRCLECI_BUILD}.  
In addition, the commit will be tagged with a release candidate tag - the package is ready for **staging**

### Creating a release

When we are happy with a release candidate, let's assume `0.0.1rc234`, we will checkout from that tag and create a
Release.  
The release should be named according to [SemVer2](https://semver.org/) rules. Please make sure that you understand them
before creating a release.

## Technical Debt

| Task | Jira Ticket|
| :---: | :---: |
| CI/CD does not support Patch version change | [OPS-143](https://deci.atlassian.net/browse/OPS-134) |  
| Connect Documentation like [this one](https://195-349373294-gh.circle-artifacts.com/0/docs/html/introduction/api.html) to be automatically mentioned | [OPS-135](https://deci.atlassian.net/browse/OPS-135) |  
| Add some test to make sure the CI flow is working | [OPS-136](https://deci.atlassian.net/browse/OPS-136) |  
| delete remote package if does not pass the tests | [OPS-137](https://deci.atlassian.net/browse/OPS-137) |
| Add PR numbers to RC versions in deci-trainer | [OPS-143](https://deci.atlassian.net/browse/OPS-143) |  
