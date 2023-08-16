import ast
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class FunctionParameter:
    name: str
    has_default: bool


@dataclass
class FunctionParameters:
    _params: List[FunctionParameter] = field(default_factory=dict)

    @property
    def all(self) -> List[str]:
        return [param.name for param in self._params]

    @property
    def required(self) -> List[str]:
        return [param.name for param in self._params if not param.has_default]

    @property
    def optional(self) -> List[str]:
        return [param.name for param in self._params if param.has_default]


@dataclass
class FunctionSignature:
    name: str
    line_num: int
    params: FunctionParameters


def parse_imports(code: str) -> Dict[str, str]:
    """Extract function signatures from the given code.

    >>> parse_imports("import package.library_v1 as library")
    {'package.library_v1': 'library'}

    >>> parse_imports("import package.library")
    {'package.library': 'package.library'}

    :param code: The Python code to analyze.
    :return:     Dictionary mapping full imported object/package name to import it's alias.
    """
    tree = ast.parse(code)
    imports = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                original_name = alias.name
                aliased_name = alias.asname if alias.asname else alias.name
                imports[original_name] = aliased_name
        elif isinstance(node, ast.ImportFrom):
            module = node.module
            for alias in node.names:
                original_name = f"{module}.{alias.name}" if module else alias.name
                aliased_name = alias.asname if alias.asname else alias.name
                imports[original_name] = aliased_name
    return imports


def parse_functions_signatures(code: str) -> Dict[str, FunctionSignature]:
    """Extract function signatures from the given Python code.

    This function returns a dictionary mapping the name of each function in the code to its signature.
    The signature includes the function name, line number, and parameters (including their names and default values).

    Example:
        >>> code = "def add(a, b=5):\\n    return a + b"
        >>> parse_functions_signatures(code)
        {
            'add': FunctionSignature(
                        name='add',
                        line_num=1,
                        params=FunctionParameters(
                            [FunctionParameter(name='a', has_default=False), FunctionParameter(name='b', has_default=True)]
                        )
                    )
        }

    :param code: The Python code to analyze.
    :return: Dictionary mapping function name to function parameters, encapsulated in a FunctionSignature object.
    """
    tree = ast.parse(code)
    signatures = {}

    # Extract top-level functions
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            signatures[node.name] = FunctionSignature(name=node.name, line_num=node.lineno, params=parse_parameters(node.args))
        # Extract methods from classes
        elif isinstance(node, ast.ClassDef):
            for method in node.body:
                if isinstance(method, ast.FunctionDef):
                    method_name = f"{node.name}.{method.name}"
                    signatures[method_name] = FunctionSignature(name=method_name, line_num=method.lineno, params=parse_parameters(method.args))

    return signatures


def parse_parameters(args: ast.arguments) -> FunctionParameters:
    """Extracts the parameters from the given args object.

    :param args:    Object from the AST (Abstract Syntax Tree).
    :return:        A FunctionParameters object representing the parameters, including their names and default values.
    """
    defaults = [None] * (len(args.args) - len(args.defaults)) + args.defaults
    parameters = FunctionParameters([FunctionParameter(name=arg.arg, has_default=default is not None) for arg, default in zip(args.args, defaults)])
    return parameters
