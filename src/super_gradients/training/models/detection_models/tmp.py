from super_gradients.training import models

decinet1_arch_params = {'ls_num_blocks': [1, 2, 6, 2],
                        'ls_block_width': [64, 136, 320, 728],
                        'ls_bottleneck_ratio': [1, 1, 1, 1],
                        'ls_group_width': [1, 1, 1, 1],
                        'se_ratio': None,
                        'stride': 2,
                        'dropout_prob': 0.0,
                        'droppath_prob': 0.0}

decinet1 = models.get(name="custom_anynet", arch_params=decinet1_arch_params, num_classes=1000)
print(decinet1)
