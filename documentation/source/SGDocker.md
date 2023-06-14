# SuperGradients Docker Container

Docker is an open-source containerization platform allowing developers to package and distribute applications in a portable and efficient way. Docker is becoming increasingly important in deep learning because it provides an easy and flexible way to manage the complex dependencies and configurations required for deep learning projects. With Docker, deep learning developers can easily package their applications and libraries into container images, which can be distributed and run on any machine with Docker installed. This simplifies the development process and makes it easier to reproduce and share deep learning experiments and results.

## Instructions and Recommended Practices

1) Follow the installation steps for the [Nvidia Docker](https://github.com/NVIDIA/nvidia-docker).
2) Pull the Docker image with the tag according to the SG version you are working with. For example, super-gradients 3.0.7:
```
docker pull deciai/super-gradients:3.0.7
```

Each SG release will push a new tag to the docker hub.
You can also use the `latest` tag:

```
docker pull deciai/super-gradients:latest
```

See the list of available tags [here](https://hub.docker.com/r/deciai/super-gradients/tags)

3) Launch the container:
```
docker run deciai/super-gradients:3.0.7
```

Recommendations for training

- For the heavier, multi-GPU training, it is best to set the shared memory to at least 64GB by appending `-shm-size=64gb` to your run command.
- Add volume mapping for your training data by appending `-v /PATH/TO/DATA_DIR/:/PATH/TO/DATA_DIR_INSIDE_THE_CONTAINER/` to your run command. Do the same for your training scripts.
- Make sure all GPUS are accessible by adding `--gpus all`.
- Run with `-it` for interactiveness.
