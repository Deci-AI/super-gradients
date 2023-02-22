from super_gradients.common.registry.registry_utils import Registry

loss_registry = Registry(items={})
register_loss = loss_registry.register
