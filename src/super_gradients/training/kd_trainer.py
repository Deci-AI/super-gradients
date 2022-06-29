from super_gradients.training.trainer import Trainer


class KDTrainer(Trainer):
    """
    Class for running SuperGradient's recipes for KD Models.
    See train_from_kd_recipe example in the examples directory to demonstrate it's usage.
    """

    @classmethod
    def build_model(cls, cfg):
        cfg.sg_model.build_model(student_architecture=cfg.student_architecture,
                                 teacher_architecture=cfg.teacher_architecture,
                                 arch_params=cfg.arch_params, student_arch_params=cfg.student_arch_params,
                                 teacher_arch_params=cfg.teacher_arch_params,
                                 checkpoint_params=cfg.checkpoint_params, run_teacher_on_eval=cfg.run_teacher_on_eval)
