import warnings

if __name__ == "__main__":

    warnings.warn("This script is deprecated and will be removed in the future. Please use `super_gradients.evaluate_checkpoint` instead.", DeprecationWarning)
    from super_gradients import evaluate_checkpoint

    evaluate_checkpoint.main()
