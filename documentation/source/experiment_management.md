# Experiment Management

## Outline
1. [Core Concepts](#core-concepts)
   - [Checkpoint Root Directory](#checkpoint-root-directory-ckpt_root_dir)
   - [Experiments](#experiments-experiment_name)
   - [Runs](#runs-run_id)
2. [File Structure of Experiments](#file-structure-of-experiments)
3. [Utilities for Experiment Management](#utilities)
   - [Get the Absolute Path of a Run Directory](#a-get-the-absolute-path-of-a-run-directory)
   - [Retrieve the Latest Run ID](#b-get-the-latest-run-id)

## Core Concepts

### Checkpoint Root Directory (`ckpt_root_dir`)
- The main directory where all experiment outputs are housed.

### Experiments (`experiment_name`)
- Symbolizes a distinct training recipe or configuration.
- Alter the `experiment_name` for transparency when updating your training recipe.
- Each training under the same `experiment_name` has its individual `run` directory, ensuring no overwrites.

### Runs (`run_id`)
- Every individual training session is termed as a `run`.
- A unique `run_id` is generated for every training, regardless of identical parameters.
- Different trainings under the same `experiment_name` maintain distinct logs and checkpoints, courtesy of their separate run directories.

## File Structure of Experiments

```
<ckpt_root_dir>
│
├── <experiment_name>
│   │
│   ├─── <run_dir>
│   │     ├─ ckpt_best.pth                   # Best performance during validation
│   │     ├─ ckpt_latest.pth                 # End of the most recent epoch
│   │     ├─ average_model.pth               # Averaged over specified epochs
│   │     ├─ ckpt_epoch_*.pth                # Checkpoints from certain epochs (e.g., epoch 10, 15)
│   │     ├─ events.out.tfevents.*           # Tensorflow run artifacts
│   │     └─ log_<timestamp>.txt             # Trainer logs of that particular run
│   │
│   └─── <other_run_dir>
│        └─ ...
│
└─── <other_experiment_name>
    │
    ├─── <run_dir>
    │     └─ ...
    │
    └─── <another_run_dir>
          └─ ...
```

## Utilities

#### A. Get the absolute path of a run directory
Manually navigate using `<ckpt_root_dir>/<experiment_name>/<run_dir>` or utilize the following programmatic approach:
```python
from super_gradients.common.environment.checkpoints_dir_utils import get_checkpoints_dir_path

checkpoints_dir_path = get_checkpoints_dir_path(experiment_name="<experiment_name>", run_id="<run_id>")
```

#### B. Get the latest run id

```python
from super_gradients.common.environment.checkpoints_dir_utils import get_latest_run_id

run_id = get_latest_run_id(experiment_name="<experiment_name>")
```
Combine with the above utility to fetch the path of the latest run directory.

**Next Steps**:
- Dive into the [checkpoints tutorial](Checkpoints.md) to grasp the essence of checkpoints, enabling you to resume trainings or access checkpoints from prior runs.
- The [logs tutorial](logs.md) focuses on the log files stored in your run directories, offering insights into the training progression.
