import warnings

if __name__ == "__main__":

    warnings.warn("This script is deprecated and will be removed in the future. Please use `super_gradients.train_from_kd_recipe` instead.", DeprecationWarning)
    from super_gradients import train_from_kd_recipe

    train_from_kd_recipe.main()
