class KDModelException(Exception):
    """Exception raised illegal training param format.

    :param desc: Explanation of the error
    """

    def __init__(self, desc: str):
        self.message = "KDTrainer: " + desc
        super().__init__(self.message)


class ArchitectureKwargsException(KDModelException):
    """Exception raised when subnet architectures are not defined."""

    def __init__(self):
        super().__init__("When architecture is not intialized both student_architecture and teacher_architecture must be passed " "through **kwargs")


class UnsupportedKDArchitectureException(KDModelException):
    """Exception raised for unsupported kd architecture.

    :param architecture: Explanation of the error
    """

    def __init__(self, architecture: str):
        super().__init__("Unsupported KD architecture: " + str(architecture))


class InconsistentParamsException(KDModelException):
    """Exception raised when values between arch_params/checkpoint_params should be equivalent.

    :param inconsistent_key1:                   Name of the key provided
    :param inconsistent_key1_container_name:    Container name of the key provided
    :param inconsistent_key2:                   Name of the key expected
    :param inconsistent_key2_container_name:    Container name of the key expected
    """

    def __init__(
        self,
        inconsistent_key1: str,
        inconsistent_key1_container_name: str,
        inconsistent_key2: str,
        inconsistent_key2_container_name: str,
    ):
        super().__init__(
            f"{inconsistent_key1} in {inconsistent_key1_container_name} must be equal to " f"{inconsistent_key2} in {inconsistent_key2_container_name}"
        )


class UnsupportedKDModelArgException(KDModelException):
    """Exception raised for unsupported args that might be supported for Trainer but not for KDTrainer.

    :param param_name: Name of the param that is not supported
    :param dict_name: Name of the dict including the param that is not supported
    """

    def __init__(self, param_name: str, dict_name: str):
        super().__init__(param_name + " in " + dict_name + " not supported for KD models.")


class TeacherKnowledgeException(KDModelException):
    """Exception raised when teacher net doesn't hold any knowledge (i.e weights are the initial ones)."""

    def __init__(self):
        super().__init__("Expected: at least one of: teacher_pretrained_weights, teacher_checkpoint_path or load_kd_trainer_checkpoint=True")


class UndefinedNumClassesException(KDModelException):
    """Exception raised when num_classes is not defined for subnets (and cannot be derived)."""

    def __init__(self):
        super().__init__("Number of classes must be defined in students and teachers arch params or by connecting to a dataset interface")
