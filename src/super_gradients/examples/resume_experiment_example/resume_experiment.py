import warnings

if __name__ == "__main__":

    warnings.warn("This script is deprecated and will be removed in the future. Please use `super_gradients.resume_experiment` instead.", DeprecationWarning)
    from super_gradients import resume_experiment

    resume_experiment.main()
