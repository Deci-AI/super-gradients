import ast
from dataclasses import dataclass, field
from typing import List, Dict, Optional


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

    def extract_methods_from_class(class_node: ast.ClassDef, derived_class_name: Optional[str] = None) -> Dict[str, FunctionSignature]:
        """
        Recursively extract methods from a class and its all its base classes.

        Given a class node from the AST, this function will extract the methods defined in the class
        as well as any methods inherited from its base classes. It produces a dictionary of method
        names (prefixed by the derived class, if specified) mapped to their `FunctionSignature` objects.


        :param class_node:              The AST node representing the class.
        :param derived_class_name :     (Optional) The name of the derived class (if any) to prefix the method names.
        :return:                        Dictionary mapping method names to their signatures.
        """

        derived_class_name = derived_class_name or class_node.name
        methods = {}

        # Extracting methods from the current class node
        for method in class_node.body:
            if isinstance(method, ast.FunctionDef):
                method_name = f"{derived_class_name}.{method.name}"
                methods[method_name] = FunctionSignature(name=method_name, line_num=method.lineno, params=parse_parameters(method.args))

        # Recursively extract methods from base classes
        for base in class_node.bases:
            if isinstance(base, ast.Name) and base.id in classes:
                base_class_methods = extract_methods_from_class(class_node=classes[base.id], derived_class_name=derived_class_name)
                methods.update(base_class_methods)

        return methods

    # Extract top-level functions and classes
    classes = {}  # Store all the classes with their node
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            signatures[node.name] = FunctionSignature(name=node.name, line_num=node.lineno, params=parse_parameters(node.args))
        elif isinstance(node, ast.ClassDef):
            classes[node.name] = node
            signatures.update(extract_methods_from_class(class_node=node))

    return signatures


def parse_parameters(args: ast.arguments) -> FunctionParameters:
    """Extracts the parameters from the given args object.

    :param args:    Object from the AST (Abstract Syntax Tree).
    :return:        A FunctionParameters object representing the parameters, including their names and default values.
    """
    defaults = [None] * (len(args.args) - len(args.defaults)) + args.defaults
    parameters = FunctionParameters([FunctionParameter(name=arg.arg, has_default=default is not None) for arg, default in zip(args.args, defaults)])
    return parameters
