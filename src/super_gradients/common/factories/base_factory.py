from typing import Union, Mapping


class AbstractFactory:

    def get(self, conf: Union[str, dict]):
        raise NotImplementedError


class BaseFactory(AbstractFactory):

    def __init__(self, type_dict: dict):
        self.type_dict = type_dict

    def get(self, conf: Union[str, dict]):
        if isinstance(conf, str):
            if conf in self.type_dict:
                return self.type_dict[conf]()
            else:
                raise RuntimeError(f"Unknown object type: {conf} in configuration. valid types are: {self.type_dict.keys()}")
        elif isinstance(conf, Mapping):
            if len(conf.keys()) > 1:
                raise RuntimeError("Malformed object definition in configuration. Expecting either a string of object type or a single entry dictionary"
                                   "{type_name(str): {parameters...}}."
                                   f"received: {conf}")

            _type = list(conf.keys())[0]
            _params = list(conf.values())[0]
            if _type in self.type_dict:
                return self.type_dict[_type](**_params)
            else:
                raise RuntimeError(f"Unknown object type: {_type} in configuration. valid types are: {self.type_dict.keys()}")
        else:
            return conf
