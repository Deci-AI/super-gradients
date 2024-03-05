"""
Entry point for training from a recipe using SuperGradients in modal environment.
General use: modal run src/super_gradients/train_from_recipe --config-name <RECIPE_CONFIG_NAME>.
For recipe's specific instructions and details refer to the recipe's configuration file in the recipes directory.
"""
import os
import dotenv
import hydra
import modal
import subprocess

from super_gradients import init_trainer

dotenv.load_dotenv()

stub = modal.Stub(name="train_from_recipe")

# git branch and repository url for the image build
# remember to have your changes pushed to the remote repository so that it can be accessed by the modal environment
git_branch = "CORE-4026_retrain_on_next_set_of_data"  # set to None to use the default branch, remember to update it after merging
repository_url = "github.com/Unstructured-IO/super-gradients-fork.git"
if git_branch:
    repository_url += f"@{git_branch}"

force_build = False  # set to True to force the image build, may be necessary not to use the cached image
gpu_count = 1  # number of GPUs to use for the modal run
gpu = modal.gpu.A100(count=gpu_count)  # A10G is time/price efficient for modal runs
data_volume = modal.Volume.persisted("od_datasets")  # will be mounted to /data in the modal environment
checkpoints_volume = modal.Volume.persisted("checkpoints")  # will be mounted to /root/modal_checkpoints in the modal environment
dotenv_secrets = modal.Secret.from_dotenv()  # loads GITHUB_TOKEN from .env file
modal_timeout = 86400  # 24 hours - maximum timeout for modal run

# double copy of recipes is a workaround for the issue with paths placement in the modal environment
# probably it can be handled in a better way
image = (
    modal.Image.from_dockerfile("Dockerfile", force_build=force_build)
    .copy_local_dir("./src/super_gradients/recipes", "/root/recipes")
    .copy_local_dir("./src/super_gradients/recipes", "/root/super_gradients/recipes")
    .copy_local_file("./src/super_gradients/launch_workaround_modal.py", "root/launch_workaround_modal.py")
    .pip_install_private_repos(repository_url, git_user=os.getenv("GITHUB_USERNAME"), secrets=[dotenv_secrets])
)


def _validate_gpu_count(modal_gpu_count, config_name):
    with hydra.initialize(config_path="recipes"):
        config = hydra.compose(config_name=config_name)
    if "training_hyperparams" in config and "num_gpus" in config.training_hyperparams:
        config_gpu_count = config.training_hyperparams.num_gpus
    else:
        config_gpu_count = 1
    if modal_gpu_count != config_gpu_count:
        raise ValueError(f"Required {config_gpu_count} GPUs for the modal run, but got {modal_gpu_count}")


@stub.function(
    image=image,
    gpu=gpu,
    volumes={"/data": data_volume, "/root/modal_checkpoints": checkpoints_volume},
    timeout=modal_timeout,
    _allow_background_volume_commits=True,
    secrets=[dotenv_secrets],
)
def _main(config_name) -> None:
    if exit_code := subprocess.call(["python", "launch_workaround_modal.py", config_name]):
        exit(exit_code)


@stub.local_entrypoint()
def main(config_name: str) -> None:
    _validate_gpu_count(gpu_count, config_name)
    init_trainer()
    _main.remote(config_name)


if __name__ == "__main__":
    main()
