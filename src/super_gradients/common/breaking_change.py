import sys
import argparse
from typing import List, Dict, Optional, Any, Union
import json
from abc import ABC
import ast
import git
from termcolor import colored
from dataclasses import dataclass, field, asdict


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

            summary += "{:<70} {:<8} {:<30} {}\n".format(
                module_path_colored, breaking_change.line_num, breaking_change.breaking_type_name, breaking_change.description
            )

        return summary

    def json(self) -> Dict[str, List[str]]:
        return asdict(self)


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


def extract_code_breaking_changes(module_path: str, source_code: str, current_code: str) -> BreakingChanges:
    """Compares two versions of code to identify breaking changes.

    :param module_path: The path to the module being compared.
    :param source_code: The source version of the code.
    :param current_code: The modified version of the code.
    :return: A BreakingChanges object detailing the differences.
    """
    breaking_changes = BreakingChanges(module_path=module_path)

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
    {'package.library_v1': 'package.library'}

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
    """Extract function signatures from the given code.

    :param code: The Python code to analyze.
    :return:     Dictionary mapping function name to function parameters.
    """
    tree = ast.parse(code)
    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    signatures = {
        function.name: FunctionSignature(name=function.name, line_num=function.lineno, params=parse_parameters(function.args)) for function in functions
    }
    return signatures


def parse_parameters(args: ast.arguments) -> FunctionParameters:
    """Extracts the parameters from the given args object.

    :param args: The arguments object from the AST.
    :return: A list of dictionaries representing the parameters.
    """
    defaults = [None] * (len(args.args) - len(args.defaults)) + args.defaults
    parameters = FunctionParameters([FunctionParameter(name=arg.arg, default=default) for arg, default in zip(args.args, defaults)])
    return parameters


def analyze_breaking_changes(verbose: bool = 1) -> List[Dict[str, Union[str, List]]]:
    """Analyze changes between the current branch (HEAD) and the master branch.
    :param verbose: If True, print the summary of breaking changes in a nicely formatted way
    :return:        List of changes, where each change is a dictionary listing each type of change for each module.
    """

    git_explorer = GitHelper(git_path="./../../..")

    summary = "{:<60} {:<8} {:<30} {}\n".format("MODULE", "LINE NO", "BREAKING TYPE", "DESCRIPTION")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", default=True, type=bool)
    parser.add_argument("--output-file", default=None, type=str)
    parser.add_argument("--fail-on-error", default=True, type=bool)
    args, _ = parser.parse_known_args()

    report = analyze_breaking_changes(verbose=args.verbose)

    if args.output_file:
        with open(args.output_file, "w") as file:
            json.dump(report, file)

    if args.fail_on_error:
        sys.exit(2)


if __name__ == "__main__":
    main()
