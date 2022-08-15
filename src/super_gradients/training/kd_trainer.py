from super_gradients.training import models

from super_gradients.training.trainer import Trainer


class KDTrainer(Trainer):
    """
    Class for running SuperGradient's recipes for KD Models.
    See train_from_kd_recipe example in the examples directory to demonstrate it's usage.
    """

    @classmethod
    def build_net_and_train(cls, cfg):
        # BUILD NETWORK
        student = models.get(architecture=cfg.student_architecture, arch_params=cfg.student_arch_params, checkpoint_params=cfg.student_checkpoint_params)
        teacher = models.get(architecture=cfg.teacher_architecture, arch_params=cfg.teacher_arch_params, checkpoint_params=cfg.teacher_checkpoint_params)

        # TRAIN
        cfg.sg_model.train(student=student,
                           teacher=teacher,
                           kd_arch_params=cfg.arch_params,
                           run_teacher_on_eval=cfg.run_teacher_on_eval,
                           training_params=cfg.training_hyperparams)
