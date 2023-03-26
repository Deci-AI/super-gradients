from enum import Enum
from typing import List

__all__ = ["Phase", "PhaseCallback", "PhaseContext", "CallbackHandler", "Callback"]


class Phase(Enum):
    PRE_TRAINING = "PRE_TRAINING"  # This event corresponds to Callback.on_training_start
    TRAIN_EPOCH_START = "TRAIN_EPOCH_START"  # This event corresponds to Callback.on_train_loader_start
    TRAIN_BATCH_END = "TRAIN_BATCH_END"  # This event corresponds to Callback.on_train_batch_loss_end
    TRAIN_BATCH_STEP = "TRAIN_BATCH_STEP"  # This event corresponds to Callback.on_train_batch_gradient_step_end
    TRAIN_EPOCH_END = "TRAIN_EPOCH_END"  # This event corresponds to Callback.on_train_loader_end
    VALIDATION_BATCH_END = "VALIDATION_BATCH_END"  # This event corresponds to Callback.on_validation_batch_end
    VALIDATION_EPOCH_END = "VALIDATION_EPOCH_END"  # This event corresponds to Callback.on_validation_loader_end
    VALIDATION_END_BEST_EPOCH = "VALIDATION_END_BEST_EPOCH"  # This event corresponds to Callback.on_validation_end_best_epoch
    TEST_BATCH_END = "TEST_BATCH_END"  # This event corresponds to Callback.on_test_batch_end
    TEST_END = "TEST_END"  # This event corresponds to Callback.on_test_loader_end
    POST_TRAINING = "POST_TRAINING"  # This event corresponds to Callback.on_training_end


class PhaseContext:
    """
    Represents the input for phase callbacks, and is constantly updated after callback calls.

    """

    def __init__(
        self,
        epoch=None,
        batch_idx=None,
        optimizer=None,
        metrics_dict=None,
        inputs=None,
        preds=None,
        target=None,
        metrics_compute_fn=None,
        loss_avg_meter=None,
        loss_log_items=None,
        criterion=None,
        device=None,
        experiment_name=None,
        ckpt_dir=None,
        net=None,
        lr_warmup_epochs=None,
        sg_logger=None,
        train_loader=None,
        valid_loader=None,
        training_params=None,
        ddp_silent_mode=None,
        checkpoint_params=None,
        architecture=None,
        arch_params=None,
        metric_idx_in_results_tuple=None,
        metric_to_watch=None,
        valid_metrics=None,
        context_methods=None,
        ema_model=None,
    ):
        self.epoch = epoch
        self.batch_idx = batch_idx
        self.optimizer = optimizer
        self.inputs = inputs
        self.preds = preds
        self.target = target
        self.metrics_dict = metrics_dict
        self.metrics_compute_fn = metrics_compute_fn
        self.loss_avg_meter = loss_avg_meter
        self.loss_log_items = loss_log_items
        self.criterion = criterion
        self.device = device
        self.stop_training = False
        self.experiment_name = experiment_name
        self.ckpt_dir = ckpt_dir
        self.net = net
        self.lr_warmup_epochs = lr_warmup_epochs
        self.sg_logger = sg_logger
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.training_params = training_params
        self.ddp_silent_mode = ddp_silent_mode
        self.checkpoint_params = checkpoint_params
        self.architecture = architecture
        self.arch_params = arch_params
        self.metric_to_watch = metric_to_watch
        self.valid_metrics = valid_metrics
        self.context_methods = context_methods
        self.ema_model = ema_model

    def update_context(self, **kwargs):
        for attr, attr_val in kwargs.items():
            setattr(self, attr, attr_val)


class Callback:
    """
    Base callback class with all the callback methods. Derived classes may override one or many of the available events
    to receive callbacks when such events are triggered by the training loop.

    The order of the events is as follows:

    on_training_start(context)                              # called once before training starts, good for setting up the warmup LR

        for epoch in range(epochs):
            on_train_loader_start(context)
                for batch in train_loader:
                    on_train_batch_start(context)
                    on_train_batch_loss_end(context)               # called after loss has been computed
                    on_train_batch_backward_end(context)           # called after .backward() was called
                    on_train_batch_gradient_step_start(context)    # called before the optimizer step about to happen (gradient clipping, logging of gradients)
                    on_train_batch_gradient_step_end(context)      # called after gradient step was done, good place to update LR (for step-based schedulers)
                    on_train_batch_end(context)
            on_train_loader_end(context)

            on_validation_loader_start(context)
                for batch in validation_loader:
                    on_validation_batch_start(context)
                    on_validation_batch_end(context)
            on_validation_loader_end(context)
            on_validation_end_best_epoch(context)

        on_test_start(context)
            for batch in test_loader:
                on_test_batch_start(context)
                on_test_batch_end(context)
        on_test_end(context)

    on_training_end(context)                    # called once after training ends.

    Correspondence mapping from the old callback API:

    on_training_start(context)                                 <-> Phase.PRE_TRAINING
    for epoch in range(epochs):
        on_train_loader_start(context)                         <-> Phase.TRAIN_EPOCH_START
            for batch in train_loader:
                on_train_batch_start(context)
                on_train_batch_loss_end(context)
                on_train_batch_backward_end(context)           <-> Phase.TRAIN_BATCH_END
                on_train_batch_gradient_step_start(context)
                on_train_batch_gradient_step_end(context)      <-> Phase.TRAIN_BATCH_STEP
                on_train_batch_end(context)
        on_train_loader_end(context)                           <-> Phase.TRAIN_EPOCH_END

        on_validation_loader_start(context)
            for batch in validation_loader:
                on_validation_batch_start(context)
                on_validation_batch_end(context)               <-> Phase.VALIDATION_BATCH_END
        on_validation_loader_end(context)                      <-> Phase.VALIDATION_EPOCH_END
        on_validation_end_best_epoch(context)                  <-> Phase.VALIDATION_END_BEST_EPOCH

    on_test_start(context)
        for batch in test_loader:
            on_test_batch_start(context)
            on_test_batch_end(context)                         <-> Phase.TEST_BATCH_END
    on_test_end(context)                                       <-> Phase.TEST_END

    on_training_end(context)                                   <-> Phase.POST_TRAINING
    """

    def on_training_start(self, context: PhaseContext) -> None:
        """
        Called once before start of the first epoch
        At this point, the context argument is guaranteed to have the following attributes:
        - optimizer
        - net
        - checkpoints_dir_path
        - criterion
        - sg_logger
        - train_loader
        - valid_loader
        - training_params
        - checkpoint_params
        - architecture
        - arch_params
        - metric_to_watch
        - device
        - ema_model

        The corresponding Phase enum value for this event is Phase.PRE_TRAINING.
        :param context:
        :return:
        """
        pass

    def on_train_loader_start(self, context: PhaseContext) -> None:
        """
        Called each epoch at the start of train data loader (before getting the first batch).
        At this point, the context argument is guaranteed to have the following attributes:
        - epoch
        The corresponding Phase enum value for this event is Phase.TRAIN_EPOCH_START.
        :param context:
        :return:
        """
        pass

    def on_train_batch_start(self, context: PhaseContext) -> None:
        """
        Called at each batch after getting batch of data from data loader and moving it to target device.
        This event triggered AFTER Trainer.pre_prediction_callback call (If it was defined).

        At this point the context argument is guaranteed to have the following attributes:
        - batch_idx
        - inputs
        - targets
        - **additional_batch_items

        :param context:
        :return:
        """
        pass

    def on_train_batch_loss_end(self, context: PhaseContext) -> None:
        """
        Called after model forward and loss computation has been done.
        At this point the context argument is guaranteed to have the following attributes:
        - preds
        - loss_log_items
        The corresponding Phase enum value for this event is Phase.TRAIN_BATCH_END.

        :param context:
        :return:
        """

        pass

    def on_train_batch_backward_end(self, context: PhaseContext) -> None:
        """
        Called after loss.backward() method was called for a given batch

        :param context:
        :return:
        """
        pass

    def on_train_batch_gradient_step_start(self, context: PhaseContext) -> None:
        """
        Called before the graadient step is about to happen.
        Good place to clip gradients (with respect to scaler), log gradients to data ratio, etc.
        :param context:
        :return:
        """
        pass

    def on_train_batch_gradient_step_end(self, context: PhaseContext) -> None:
        """
        Called after gradient step has been performed. Good place to update LR (for step-based schedulers)
        The corresponding Phase enum value for this event is Phase.TRAIN_BATCH_STEP.
        :param context:
        :return:
        """
        pass

    def on_train_batch_end(self, context: PhaseContext) -> None:
        """
        Called after all forward/backward/optimizer steps have been performed for a given batch and there is nothing left to do.

        :param context:
        :return:
        """

        pass

    def on_train_loader_end(self, context: PhaseContext) -> None:
        """
        Called each epoch at the end of train data loader (after processing the last batch).
        The corresponding Phase enum value for this event is Phase.TRAIN_EPOCH_END.
        :param context:
        :return:
        """

        pass

    def on_validation_loader_start(self, context: PhaseContext) -> None:
        """
        Called each epoch at the start of validation data loader (before getting the first batch).
        :param context:
        :return:
        """

        pass

    def on_validation_batch_start(self, context: PhaseContext) -> None:
        """
        Called at each batch after getting batch of data from validation loader and moving it to target device.
        :param context:
        :return:
        """
        pass

    def on_validation_batch_end(self, context: PhaseContext) -> None:
        """
        Called after all forward step / loss / metric computation have been performed for a given batch and there is nothing left to do.
        The corresponding Phase enum value for this event is Phase.VALIDATION_BATCH_END.
        :param context:
        :return:
        """
        pass

    def on_validation_loader_end(self, context: PhaseContext) -> None:
        """
        Called each epoch at the end of validation data loader (after processing the last batch).
        The corresponding Phase enum value for this event is Phase.VALIDATION_EPOCH_END.
        :param context:
        :return:
        """
        pass

    def on_validation_end_best_epoch(self, context: PhaseContext) -> None:
        """
        Called each epoch after validation has been performed and the best metric has been achieved.
        The corresponding Phase enum value for this event is Phase.VALIDATION_END_BEST_EPOCH.
        :param context:
        :return:
        """
        pass

    def on_test_loader_start(self, context: PhaseContext) -> None:
        """
        Called once at the start of test data loader (before getting the first batch).
        :param context:
        :return:
        """

        pass

    def on_test_batch_start(self, context: PhaseContext) -> None:
        """
        Called at each batch after getting batch of data from test loader and moving it to target device.
        :param context:
        :return:
        """
        pass

    def on_test_batch_end(self, context: PhaseContext) -> None:
        """
        Called after all forward step have been performed for a given batch and there is nothing left to do.
        The corresponding Phase enum value for this event is Phase.TEST_BATCH_END.
        :param context:
        :return:
        """
        pass

    def on_test_loader_end(self, context: PhaseContext) -> None:
        """
        Called once at the end of test data loader (after processing the last batch).
        The corresponding Phase enum value for this event is Phase.TEST_END.
        :param context:
        :return:
        """
        pass

    def on_training_end(self, context: PhaseContext) -> None:
        """
        Called once after the training loop has finished (Due to reaching optimization criterion or because of an error.)
        The corresponding Phase enum value for this event is Phase.POST_TRAINING.
        :param context:
        :return:
        """
        pass


class PhaseCallback(Callback):
    """
    Kept here to keep backward compatibility with old code. New callbacks should use Callback class instead.
    This callback supports receiving only a subset of events defined in Phase enum:

    PRE_TRAINING = "PRE_TRAINING"
    TRAIN_EPOCH_START = "TRAIN_EPOCH_START"
    TRAIN_BATCH_END = "TRAIN_BATCH_END"
    TRAIN_BATCH_STEP = "TRAIN_BATCH_STEP"
    TRAIN_EPOCH_END = "TRAIN_EPOCH_END"

    VALIDATION_BATCH_END = "VALIDATION_BATCH_END"
    VALIDATION_EPOCH_END = "VALIDATION_EPOCH_END"
    VALIDATION_END_BEST_EPOCH = "VALIDATION_END_BEST_EPOCH"

    TEST_BATCH_END = "TEST_BATCH_END"
    TEST_END = "TEST_END"
    POST_TRAINING = "POST_TRAINING"
    """

    def __init__(self, phase: Phase):
        self.phase = phase

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.__class__.__name__

    def on_training_start(self, context: PhaseContext) -> None:
        if self.phase == Phase.PRE_TRAINING:
            self(context)

    def on_train_loader_start(self, context: PhaseContext) -> None:
        if self.phase == Phase.TRAIN_EPOCH_START:
            self(context)

    def on_train_batch_loss_end(self, context: PhaseContext) -> None:
        if self.phase == Phase.TRAIN_BATCH_END:
            self(context)

    def on_train_batch_gradient_step_end(self, context: PhaseContext) -> None:
        if self.phase == Phase.TRAIN_BATCH_STEP:
            self(context)

    def on_train_loader_end(self, context: PhaseContext) -> None:
        if self.phase == Phase.TRAIN_EPOCH_END:
            self(context)

    def on_validation_batch_end(self, context: PhaseContext) -> None:
        if self.phase == Phase.VALIDATION_BATCH_END:
            self(context)

    def on_validation_loader_end(self, context: PhaseContext) -> None:
        if self.phase == Phase.VALIDATION_EPOCH_END:
            self(context)

    def on_validation_end_best_epoch(self, context: PhaseContext) -> None:
        if self.phase == Phase.VALIDATION_END_BEST_EPOCH:
            self(context)

    def on_test_batch_end(self, context: PhaseContext) -> None:
        if self.phase == Phase.TEST_BATCH_END:
            self(context)

    def on_test_loader_end(self, context: PhaseContext) -> None:
        if self.phase == Phase.TEST_END:
            self(context)

    def on_training_end(self, context: PhaseContext) -> None:
        if self.phase == Phase.POST_TRAINING:
            self(context)


class CallbackHandler(Callback):
    """
    Runs all callbacks

    :param callbacks: Callbacks to be run.
    """

    def __init__(self, callbacks: List[Callback]):
        # TODO: Add reordering of callbacks to make sure that they are called in the right order
        # For instance, two callbacks may be dependent on each other, so the first one should be called first
        # Example: Gradient Clipping & Gradient Logging callback. We first need to clip the gradients, and then log them
        # So if user added them in wrong order we can guarantee their order would be correct.
        # We can achieve this by adding a property to the callback to the callback indicating it's priority:
        # Forward   = 0
        # Loss      = 100
        # Backward  = 200
        # Metrics   = 300
        # Scheduler = 400
        # Logging   = 500
        # So ordering callbacks by their order would ensure than we first run all Forward-related callbacks (for a given event),
        # Than backward, and only then - logging.
        self.callbacks = callbacks

    def on_training_start(self, context: PhaseContext) -> None:
        for callback in self.callbacks:
            callback.on_training_start(context)

    def on_train_loader_start(self, context: PhaseContext) -> None:
        for callback in self.callbacks:
            callback.on_train_loader_start(context)

    def on_train_batch_start(self, context: PhaseContext) -> None:
        for callback in self.callbacks:
            callback.on_train_batch_start(context)

    def on_train_batch_loss_end(self, context: PhaseContext) -> None:
        for callback in self.callbacks:
            callback.on_train_batch_loss_end(context)

    def on_train_batch_backward_end(self, context: PhaseContext) -> None:
        for callback in self.callbacks:
            callback.on_train_batch_backward_end(context)

    def on_train_batch_gradient_step_start(self, context: PhaseContext) -> None:
        for callback in self.callbacks:
            callback.on_train_batch_gradient_step_start(context)

    def on_train_batch_gradient_step_end(self, context: PhaseContext) -> None:
        for callback in self.callbacks:
            callback.on_train_batch_gradient_step_end(context)

    def on_train_batch_end(self, context: PhaseContext) -> None:
        for callback in self.callbacks:
            callback.on_train_batch_end(context)

    def on_validation_loader_start(self, context: PhaseContext) -> None:
        for callback in self.callbacks:
            callback.on_validation_loader_start(context)

    def on_validation_batch_start(self, context: PhaseContext) -> None:
        for callback in self.callbacks:
            callback.on_validation_batch_start(context)

    def on_validation_batch_end(self, context: PhaseContext) -> None:
        for callback in self.callbacks:
            callback.on_validation_batch_end(context)

    def on_validation_loader_end(self, context: PhaseContext) -> None:
        for callback in self.callbacks:
            callback.on_validation_loader_end(context)

    def on_train_loader_end(self, context: PhaseContext) -> None:
        for callback in self.callbacks:
            callback.on_train_loader_end(context)

    def on_training_end(self, context: PhaseContext) -> None:
        for callback in self.callbacks:
            callback.on_training_end(context)

    def on_validation_end_best_epoch(self, context: PhaseContext) -> None:
        for callback in self.callbacks:
            callback.on_validation_end_best_epoch(context)

    def on_test_loader_start(self, context: PhaseContext) -> None:
        for callback in self.callbacks:
            callback.on_test_loader_start(context)

    def on_test_batch_start(self, context: PhaseContext) -> None:
        for callback in self.callbacks:
            callback.on_test_batch_start(context)

    def on_test_batch_end(self, context: PhaseContext) -> None:
        for callback in self.callbacks:
            callback.on_test_batch_end(context)

    def on_test_loader_end(self, context: PhaseContext) -> None:
        for callback in self.callbacks:
            callback.on_test_loader_end(context)
