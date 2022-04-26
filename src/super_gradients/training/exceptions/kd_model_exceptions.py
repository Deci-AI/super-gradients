class KDModelException(Exception):
    """Exception raised illegal training param format.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, desc):
        self.message = "KDModel: " + desc
        super().__init__(self.message)


class ArchitectureKwargsException(KDModelException):
    """Exception raised when subnet architectures are not defined.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self):
        super().__init__(
            "When architecture is not intialized both student_architecture and teacher_architecture must be passed "
            "through **kwargs")


class UnsupportedKDArchitectureException(KDModelException):
    """Exception raised for unsupported kd architecture.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, architecture):
        super().__init__("Unsupported KD architecture: " + str(architecture))


class InconsistentParamsException(KDModelException):
    """Exception raised when values between arch_params/checkpoint_params should be equivalent.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, inconsistent_key1: str, inconsistent_key1_container_name: str, inconsistent_key2: str,
                 inconsistent_key2_container_name: str, ):
        super().__init__(
            inconsistent_key1 + " in " + inconsistent_key1_container_name + " must be equal to " + inconsistent_key2 + " in " + inconsistent_key2_container_name)


class UnsupportedKDModelArgException(KDModelException):
    """Exception raised for unsupported args that might be supported for SgModel but not for KDModel.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, param_name: str, dict_name: str):
        super().__init__(
            param_name + " in " + dict_name + " not supported for KD models.")


class TeacherKnowledgeException(KDModelException):
    """Exception raised when teacher net doesn't hold any knowledge (i.e weights are the initial ones).

    Attributes:
        message -- explanation of the error
    """

    def __init__(self):
        super().__init__(
            "Expected: at least one of: teacher_pretrained_weights, teacher_checkpoint_path or load_kd_model_checkpoint=True")


class UndefinedNumClassesException(KDModelException):
    """Exception raised when num_classes is not defined for subnets (and cannot be derived).

    Attributes:
        message -- explanation of the error
    """
    def __init__(self):
        super().__init__(
            'Number of classes must be defined in students and teachers arch params or by connecting to a dataset interface')
