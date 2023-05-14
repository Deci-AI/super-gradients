import warnings

if __name__ == "__main__":

    warnings.warn("This script is deprecated and will be removed in the future. Please use `super_gradients.qat_from_recipe` instead.", DeprecationWarning)
    from super_gradients import qat_from_recipe

    qat_from_recipe.main()
