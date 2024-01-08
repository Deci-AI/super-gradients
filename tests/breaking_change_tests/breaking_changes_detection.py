import ast
from abc import ABC
from pathlib import Path
from typing import List, Dict, Union

from termcolor import colored
from dataclasses import dataclass, field, asdict

from .code_parser import parse_functions_signatures, parse_imports


MODULE_PATH_COLOR = "yellow"
SOURCE_CODE_COLOR = "blue"
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

    @property
    def is_empty(self) -> bool:
        return len(self.classes_removed + self.imports_removed + self.functions_removed + self.params_removed + self.required_params_added) == 0


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

    # IMPORTS - Check import ONLY if __init__ file and ignores non-SG imports.
    current_imports = parse_imports(code=current_code)
    if module_path.endswith("__init__.py"):
        source_imports = parse_imports(code=source_code)
        breaking_changes.imports_removed = [
            ImportRemoved(import_name=source_import, line_num=0)
            for source_import in source_imports
            if (source_import not in current_imports) and ("super_gradients" in source_import)
        ]

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
            # Count a function as removed only if it was removed AND it was not added in the imports!
            imported_function_names = current_imports.values()
            if function_name not in imported_function_names:
                breaking_changes.functions_removed.append(
                    FunctionRemoved(
                        function_name=function_name,
                        line_num=source_function_signature.line_num,
                    )
                )

    return breaking_changes


def analyze_breaking_changes(verbose: bool = 1, source_branch: str = "master") -> List[Dict[str, Union[str, List]]]:
    """Analyze changes between the current branch (HEAD) and the master branch.
    :param verbose:         If True, print the summary of breaking changes in a nicely formatted way
    :param source_branch:   The branch source branch, to which we will compare the HEAD.
    :return:        List of changes, where each change is a dictionary listing each type of change for each module.
    """
    print("\n" + "=" * 50)
    print(f"Analyzing breaking changes, comparing `HEAD` to `{source_branch}`...")
    # GitHelper requires `git` library which should NOT be required for the other functions
    from .git_utils import GitHelper

    root_dir = str(Path(__file__).resolve().parents[2])
    git_explorer = GitHelper(git_path=root_dir)

    changed_sg_modules = [
        module_path
        for module_path in git_explorer.diff_files(source_branch=source_branch, current_branch="HEAD")
        if module_path.startswith("src/super_gradients/") and not module_path.startswith("src/super_gradients/examples/")
    ]

    summary = ""
    breaking_changes_list = []
    for module_path in changed_sg_modules:

        master_code = git_explorer.load_branch_file(branch=source_branch, file_path=module_path)
        head_code = git_explorer.load_branch_file(branch="HEAD", file_path=module_path)
        breaking_changes = extract_code_breaking_changes(module_path=module_path, source_code=master_code, current_code=head_code)

        if not breaking_changes.is_empty:
            breaking_changes_list.append(breaking_changes.json())
        summary += str(breaking_changes)

    if verbose:
        if summary:
            print("{:<60} {:<8} {:<30} {}\n".format("MODULE", "LINE NO", "BREAKING TYPE", "DESCRIPTION (Master -> HEAD)"))
            print("-" * 175 + "\n")
            print(summary)
        else:
            print(colored("NO BREAKING CHANGE DETECTED!", "green"))

    return breaking_changes_list
