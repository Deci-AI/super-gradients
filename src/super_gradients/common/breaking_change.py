import sys
import os
import argparse
from typing import List, Dict, Union
import json
from abc import ABC
import ast
import git
from termcolor import colored
from dataclasses import dataclass, field, asdict


MODULE_PATH_COLOR = "yellow"
SOURCE_CODE_COLOR = "blue"
BREAKING_OBJECT_COLOR = "red"


class GitHelper:
    def __init__(self, git_path: str):
        self.repo = git.Repo(git_path)

    def diff_files(self, source_branch: str, current_branch: str) -> List[str]:
        source_commit = self.repo.commit(source_branch)
        current_commit = self.repo.commit(current_branch)
        return [diff.a_path for diff in source_commit.diff(current_commit) if ".py" in diff.a_path]

    def load_branch_file(self, branch: str, file_path: str) -> str:
        tree = self.repo.commit(branch).tree

        try:  # It looks like there is no simple way to check if a file exists in the tree... So we directly check with try/except
            return tree[file_path].data_stream.read()
        except KeyError:
            return ""


@dataclass
class AbstractBreakingChange(ABC):
    line_num: int

    @property
    def description(self) -> str:
        raise NotImplementedError()

    @property
    def breaking_type_name(self) -> str:
        raise NotImplementedError()


@dataclass
class ClassRemoved(AbstractBreakingChange):
    class_name: str
    line_num: int

    @property
    def description(self) -> str:
        return f"{colored(self.class_name, SOURCE_CODE_COLOR)} -> {colored('X', BREAKING_OBJECT_COLOR)}"

    @property
    def breaking_type_name(self) -> str:
        return "CLASS REMOVED"


@dataclass
class ImportRemoved(AbstractBreakingChange):
    import_name: str
    line_num: int

    @property
    def description(self) -> str:
        return f"{colored(self.import_name, SOURCE_CODE_COLOR)} -> {colored('X', BREAKING_OBJECT_COLOR)}"

    @property
    def breaking_type_name(self) -> str:
        return "IMPORT REMOVED"


@dataclass
class FunctionRemoved(AbstractBreakingChange):
    function_name: str
    line_num: int

    @property
    def description(self) -> str:
        return f"{colored(self.function_name, SOURCE_CODE_COLOR)} -> {colored('X', BREAKING_OBJECT_COLOR)}"

    @property
    def breaking_type_name(self) -> str:
        return "FUNCTION REMOVED"


@dataclass
class ParameterRemoved(AbstractBreakingChange):
    parameter_name: str
    function_name: str
    line_num: int

    @property
    def description(self) -> str:
        source_fn_colored = colored(self.function_name, SOURCE_CODE_COLOR)
        current_fn_colored = colored(self.function_name, "yellow")
        param_colored = colored(self.parameter_name, BREAKING_OBJECT_COLOR)
        return f"{source_fn_colored}(..., {param_colored}) -> {current_fn_colored}(...)"

    @property
    def breaking_type_name(self) -> str:
        return "FUNCTION PARAMETER REMOVED"


@dataclass
class RequiredParameterAdded(AbstractBreakingChange):
    parameter_name: str
    function_name: str
    line_num: int

    @property
    def description(self) -> str:
        source_fn_colored = colored(self.function_name, SOURCE_CODE_COLOR)
        current_fn_colored = colored(self.function_name, "yellow")
        param_colored = colored(self.parameter_name, BREAKING_OBJECT_COLOR)
        # fn_colored = colored(self.function_name, 'yellow')
        # param_colored = colored(self.parameter_name, BREAKING_OBJECT_COLOR)
        return f"{source_fn_colored}(...) -> {current_fn_colored}(..., {param_colored})"

    @property
    def breaking_type_name(self) -> str:
        return "FUNCTION PARAMETER ADDED"


@dataclass
class BreakingChanges:
    module_path: str
    classes_removed: List[ClassRemoved] = field(default_factory=list)
    imports_removed: List[ImportRemoved] = field(default_factory=list)
    functions_removed: List[FunctionRemoved] = field(default_factory=list)
    params_removed: List[ParameterRemoved] = field(default_factory=list)
    required_params_added: List[RequiredParameterAdded] = field(default_factory=list)

    def __str__(self) -> str:
        summary = ""
        module_path_colored = colored(self.module_path, MODULE_PATH_COLOR)

        breaking_changes: List[AbstractBreakingChange] = (
            self.classes_removed + self.imports_removed + self.functions_removed + self.params_removed + self.required_params_added
        )
        for breaking_change in breaking_changes:

            summary += "{:<70} {:<8} {:<30} {}\n".format(
                module_path_colored, breaking_change.line_num, breaking_change.breaking_type_name, breaking_change.description
            )

        return summary

    def json(self) -> Dict[str, List[str]]:
        return asdict(self)


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


def extract_code_breaking_changes(module_path: str, source_code: str, current_code: str) -> BreakingChanges:
    """Compares two versions of code to identify breaking changes.

    :param module_path: The path to the module being compared.
    :param source_code: The source version of the code.
    :param current_code: The modified version of the code.
    :return: A BreakingChanges object detailing the differences.
    """
    breaking_changes = BreakingChanges(module_path=module_path)

    source_classes = {node.name: node for node in ast.walk(ast.parse(source_code)) if isinstance(node, ast.ClassDef)}
    current_classes = {node.name: node for node in ast.walk(ast.parse(current_code)) if isinstance(node, ast.ClassDef)}

    # ClassRemoved
    for class_name, source_class in source_classes.items():
        if class_name not in current_classes:
            breaking_changes.classes_removed.append(
                ClassRemoved(
                    class_name=class_name,
                    line_num=source_class.lineno,
                )
            )

    # FUNCTION SIGNATURES
    source_functions_signatures = parse_functions_signatures(source_code)
    current_functions_signatures = parse_functions_signatures(current_code)
    for function_name, source_function_signature in source_functions_signatures.items():

        if function_name in current_functions_signatures:
            current_function_signature = current_functions_signatures[function_name]

            # ParameterRemoved
            for source_param in source_function_signature.params.all:
                if source_param not in current_function_signature.params.all:
                    breaking_changes.params_removed.append(
                        ParameterRemoved(
                            function_name=function_name,
                            parameter_name=source_param,
                            line_num=current_function_signature.line_num,
                        )
                    )

            # RequiredParameterAdded
            for current_param in current_function_signature.params.required:
                if current_param not in source_function_signature.params.required:
                    breaking_changes.required_params_added.append(
                        RequiredParameterAdded(
                            function_name=function_name,
                            parameter_name=current_param,
                            line_num=current_function_signature.line_num,
                        )
                    )

        else:
            # FunctionRemoved
            breaking_changes.functions_removed.append(
                FunctionRemoved(
                    function_name=function_name,
                    line_num=source_function_signature.line_num,
                )
            )

    # Check import ONLY if __init__ file.
    if module_path.endswith("__init__.py"):
        source_imports = parse_imports(code=source_code)
        current_imports = parse_imports(code=current_code)
        breaking_changes.imports_removed = [
            ImportRemoved(import_name=source_import, line_num=0) for source_import in source_imports if source_import not in current_imports
        ]
    return breaking_changes


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


def analyze_breaking_changes(verbose: bool = 1) -> List[Dict[str, Union[str, List]]]:
    """Analyze changes between the current branch (HEAD) and the master branch.
    :param verbose: If True, print the summary of breaking changes in a nicely formatted way
    :return:        List of changes, where each change is a dictionary listing each type of change for each module.
    """

    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    git_explorer = GitHelper(git_path=root_dir)

    summary = "{:<60} {:<8} {:<30} {}\n".format("MODULE", "LINE NO", "BREAKING TYPE", "DESCRIPTION (Master -> HEAD)")
    summary += "-" * 175 + "\n"

    report = []
    for module_path in git_explorer.diff_files(source_branch="master", current_branch="HEAD"):

        master_code = git_explorer.load_branch_file(branch="master", file_path=module_path)
        head_code = git_explorer.load_branch_file(branch="HEAD", file_path=module_path)
        breaking_changes = extract_code_breaking_changes(module_path=module_path, source_code=master_code, current_code=head_code)

        report.append(breaking_changes.json())
        summary += str(breaking_changes)

    if verbose:
        if report:
            print(summary)
        else:
            print(colored("NO BREAKING CHANGE DETECTED!", "green"))

    return report


def main():
    parser = argparse.ArgumentParser(description="Example script using flags")
    parser.add_argument("--verbose", action="store_true", default=True, help="Enable verbose mode")
    parser.add_argument("--output-file", default=None, type=str, help="Output file name")
    parser.add_argument("--fail-on-error", action="store_true", default=True, help="Fail on error")

    args = parser.parse_args()

    report = analyze_breaking_changes(verbose=args.verbose)

    if args.output_file:
        with open(args.output_file, "w") as file:
            json.dump(report, file)

    if args.fail_on_error:
        sys.exit(2)


if __name__ == "__main__":
    main()
