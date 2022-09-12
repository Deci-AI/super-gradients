from super_gradients import init_trainer
from super_gradients.common import MultiGPUMode
from super_gradients.training.utils.distributed_training_utils import setup_gpu_mode

# Initialize the environment
init_trainer()

# Launch DDP
setup_gpu_mode(gpu_mode=MultiGPUMode.DISTRIBUTED_DATA_PARALLEL, num_gpus=4)

# Define the classes
trainer = ...
model = ...
train_dataloader = ...
val_dataloader = ...
training_params = ...

# The trainer will run on DDP without anything else to change
trainer.train(model=model,
              train_loader=train_dataloader,
              valid_loader=val_dataloader,
              training_params=training_params)
