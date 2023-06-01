import os
import torch
import numpy as np
from super_gradients.training.utils.checkpoint_utils import read_ckpt_state_dict
from super_gradients.training.utils.utils import move_state_dict_to_device


class ModelWeightAveraging:
    """
    Utils class for managing the averaging of the best several snapshots into a single model.
    A snapshot dictionary file and the average model will be saved / updated at every epoch and evaluated only when
    training is completed. The snapshot file will only be deleted upon completing the training.
    The snapshot dict will be managed on cpu.
    """

    def __init__(
        self,
        ckpt_dir,
        greater_is_better,
        metric_to_watch="acc",
        metric_idx=1,
        load_checkpoint=False,
        number_of_models_to_average=10,
    ):
        """
        Init the ModelWeightAveraging
        :param ckpt_dir: the directory where the checkpoints are saved
        :param metric_to_watch: monitoring loss or acc, will be identical to that which determines best_model
        :param metric_idx:
        :param load_checkpoint: whether to load pre-existing snapshot dict.
        :param number_of_models_to_average: number of models to average
        """

        self.averaging_snapshots_file = os.path.join(ckpt_dir, "averaging_snapshots.pkl")
        self.number_of_models_to_average = number_of_models_to_average
        self.metric_to_watch = metric_to_watch
        self.metric_idx = metric_idx
        self.greater_is_better = greater_is_better

        # if continuing training, copy previous snapshot dict if exist
        if load_checkpoint and ckpt_dir is not None and os.path.isfile(self.averaging_snapshots_file):
            averaging_snapshots_dict = read_ckpt_state_dict(self.averaging_snapshots_file)

        else:
            averaging_snapshots_dict = {"snapshot" + str(i): None for i in range(self.number_of_models_to_average)}
            # if metric to watch is acc, hold a zero array, if loss hold inf array
            if self.greater_is_better:
                averaging_snapshots_dict["snapshots_metric"] = -1 * np.inf * np.ones(self.number_of_models_to_average)
            else:
                averaging_snapshots_dict["snapshots_metric"] = np.inf * np.ones(self.number_of_models_to_average)

            torch.save(averaging_snapshots_dict, self.averaging_snapshots_file)

    def update_snapshots_dict(self, model, validation_results_tuple):
        """
        Update the snapshot dict and returns the updated average model for saving
        :param model: the latest model
        :param validation_results_tuple: performance of the latest model
        """
        averaging_snapshots_dict = self._get_averaging_snapshots_dict()

        # IF CURRENT MODEL IS BETTER, TAKING HIS PLACE IN ACC LIST AND OVERWRITE THE NEW AVERAGE
        require_update, update_ind = self._is_better(averaging_snapshots_dict, validation_results_tuple)
        if require_update:
            # moving state dict to cpu
            new_sd = model.state_dict()
            new_sd = move_state_dict_to_device(new_sd, "cpu")

            averaging_snapshots_dict["snapshot" + str(update_ind)] = new_sd
            averaging_snapshots_dict["snapshots_metric"][update_ind] = validation_results_tuple[self.metric_idx]

        return averaging_snapshots_dict

    def get_average_model(self, model, validation_results_tuple=None):
        """
        Returns the averaged model
        :param model: will be used to determine arch
        :param validation_results_tuple: if provided, will update the average model before returning
        :param target_device: if provided, return sd on target device

        """
        # If validation tuple is provided, update the average model
        if validation_results_tuple is not None:
            averaging_snapshots_dict = self.update_snapshots_dict(model, validation_results_tuple)
        else:
            averaging_snapshots_dict = self._get_averaging_snapshots_dict()

        torch.save(averaging_snapshots_dict, self.averaging_snapshots_file)
        average_model_sd = averaging_snapshots_dict["snapshot0"]
        for n_model in range(1, self.number_of_models_to_average):
            if averaging_snapshots_dict["snapshot" + str(n_model)] is not None:
                net_sd = averaging_snapshots_dict["snapshot" + str(n_model)]
                # USING MOVING AVERAGE
                for key in average_model_sd:
                    average_model_sd[key] = torch.true_divide(average_model_sd[key] * n_model + net_sd[key], (n_model + 1))

        return average_model_sd

    def cleanup(self):
        """
        Delete snapshot file when reaching the last epoch
        """
        os.remove(self.averaging_snapshots_file)

    def _is_better(self, averaging_snapshots_dict, validation_results_tuple):
        """
        Determines if the new model is better according to the specified metrics
        :param averaging_snapshots_dict: snapshot dict
        :param validation_results_tuple: latest model performance
        """
        snapshot_metric_array = averaging_snapshots_dict["snapshots_metric"]
        val = validation_results_tuple[self.metric_idx]

        if self.greater_is_better:
            update_ind = np.argmin(snapshot_metric_array)
        else:
            update_ind = np.argmax(snapshot_metric_array)

        if (self.greater_is_better and val > snapshot_metric_array[update_ind]) or (not self.greater_is_better and val < snapshot_metric_array[update_ind]):
            return True, update_ind

        return False, None

    def _get_averaging_snapshots_dict(self):
        return torch.load(self.averaging_snapshots_file)
