from typing import List, Dict, Optional, Any
import json
from termcolor import colored
from dataclasses import dataclass, field, asdict

import ast
import git
from abc import ABC

module_path_COLOR = "yellow"
BREAKING_TYPE_COLOR = "blue"
BREAKING_OBJECT_COLOR = "red"


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
class ImportRemoved(AbstractBreakingChange):
    import_name: str
    line_num: int

    @property
    def description(self) -> str:
        return f"{colored(self.import_name, BREAKING_OBJECT_COLOR)} was removed from module"

    @property
    def breaking_type_name(self) -> str:
        return "IMPORT REMOVED"


@dataclass
class FunctionRemoved(AbstractBreakingChange):
    function_name: str
    line_num: int

    @property
    def description(self) -> str:
        return f"{colored(self.function_name, BREAKING_OBJECT_COLOR)} was removed from module"

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
        return f"{colored(self.parameter_name, BREAKING_OBJECT_COLOR)} removed from function {colored(self.function_name, 'yellow')}"

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
        return f"{colored(self.parameter_name, BREAKING_OBJECT_COLOR)} was added to function {colored(self.function_name, 'yellow')}"

    @property
    def breaking_type_name(self) -> str:
        return "FUNCTION PARAMETER ADDED"


@dataclass
class BreakingChanges:
    module_path: str
    imports_removed: List[ImportRemoved] = field(default_factory=list)
    functions_removed: List[FunctionRemoved] = field(default_factory=list)
    params_removed: List[ParameterRemoved] = field(default_factory=list)
    required_params_added: List[RequiredParameterAdded] = field(default_factory=list)

    def __str__(self) -> str:
        summary = ""
        module_path_colored = colored(self.module_path, module_path_COLOR)

        breaking_changes: List[AbstractBreakingChange] = self.imports_removed + self.functions_removed + self.params_removed + self.required_params_added
        for breaking_change in breaking_changes:

            summary += "{:<75} {:<5} {:<75} {}".format(
                module_path_colored, breaking_change.line_num, breaking_change.breaking_type_name, breaking_change.description
            )

        return summary

    def json(self) -> Dict[str, List[str]]:
        return asdict(self)


class GitHelper:
    def __init__(self):
        self.repo = git.Repo("./../../..")

    def current_files(self, source_branch: str, current_branch: str) -> List[str]:
        source_commit = self.repo.commit(source_branch)
        current_commit = self.repo.commit(current_branch)
        return [diff.a_path for diff in source_commit.diff(current_commit) if ".py" in diff.a_path]

    def load_branch_file(self, branch: str, file_path: str) -> str:
        tree = self.repo.commit(branch).tree

        try:  # It looks like there is no simple way to check if a file exists in the tree... So we check with try/except
            return tree[file_path].data_stream.read()
        except KeyError:
            return ""


@dataclass
class FunctionParameter:
    name: str
    default: Optional[Any] = None


@dataclass
class FunctionParameters:
    _params: List[FunctionParameter] = field(default_factory=dict)

    @property
    def all(self) -> List[str]:
        return [param.name for param in self._params]

    @property
    def required(self) -> List[str]:
        return [param.name for param in self._params if param.default is None]

    @property
    def optional(self) -> List[str]:
        return [param.name for param, value in self._params if param.default is not None]


@dataclass
class FunctionSignature:
    name: str
    line_num: int
    params: FunctionParameters


def get_imports(code: str) -> Dict[str, str]:
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


def compare_code(module_path: str, source_code: str, current_code: str) -> BreakingChanges:
    """Compares two versions of code to identify breaking changes.

    :param module_path: The path to the module being compared.
    :param source_code: The source version of the code.
    :param current_code: The modified version of the code.
    :return: A BreakingChanges object detailing the differences.
    """
    breaking_changes = BreakingChanges(module_path=module_path)

    # FUNCTION SIGNATURES
    source_functions_signatures = extract_signatures(source_code)
    current_functions_signatures = extract_signatures(current_code)
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
        source_imports = get_imports(code=source_code)
        current_imports = get_imports(code=current_code)
        breaking_changes.imports_removed = [
            ImportRemoved(import_name=source_import, line_num=0) for source_import in source_imports if source_import not in current_imports
        ]
    return breaking_changes


def extract_signatures(code: str) -> Dict[str, FunctionSignature]:
    """Extracts function signatures from the given code.

    :param code: The Python code to analyze.
    :return:     Dictionary mapping function name to function parameters.
    """
    tree = ast.parse(code)
    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    signatures = {
        function.name: FunctionSignature(name=function.name, line_num=function.lineno, params=extract_parameters(function.args)) for function in functions
    }
    return signatures


def extract_parameters(args: ast.arguments) -> FunctionParameters:
    """Extracts the parameters from the given args object.

    :param args: The arguments object from the AST.
    :return: A list of dictionaries representing the parameters.
    """
    defaults = [None] * (len(args.args) - len(args.defaults)) + args.defaults
    parameters = FunctionParameters([FunctionParameter(name=arg.arg, default=default) for arg, default in zip(args.args, defaults)])
    return parameters


def main():
    git_explorer = GitHelper()

    reports = []
    for module_path in git_explorer.current_files(source_branch="master", current_branch="HEAD"):

        master_code = git_explorer.load_branch_file(branch="master", file_path=module_path)
        head_code = git_explorer.load_branch_file(branch="HEAD", file_path=module_path)
        breaking_changes = compare_code(module_path=module_path, source_code=master_code, current_code=head_code)

        reports.append(breaking_changes.json())
        print(str(breaking_changes))

    with open("report.json", "w") as file:
        json.dump(reports, file)


# TODO: change - fail import only oif init - add line number

if __name__ == "__main__":
    main()
