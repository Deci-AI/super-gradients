import ast
import sys
from typing import Dict, List
from pathlib import Path


def get_changed_files(file_path: str) -> List[str]:
    with open(file_path, "r") as file:
        return file.readlines()


def get_file_content(file_path: str) -> str:
    with open(file_path, "r") as file:
        return file.read()


def get_function_signatures(code: str) -> Dict[str, List[str]]:
    tree = ast.parse(code)
    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    signatures = {}
    for function in functions:
        name = function.name
        args = [arg.arg for arg in function.args.args]
        signatures[name] = args
    return signatures


# def get_imports(code: str) -> List[str]:
#     tree = ast.parse(code)
#     return [node.name for node in ast.walk(tree) if isinstance(node, ast.Import)]
def get_imports(code: str) -> List[str]:
    tree = ast.parse(code)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module
            if module:
                imports.extend(f"{module}.{alias.name}" for alias in node.names)
            else:
                imports.extend(alias.name for alias in node.names)
    return imports


def compare_signatures(old_code: str, new_code: str) -> None:
    old_signatures = get_function_signatures(old_code)
    new_signatures = get_function_signatures(new_code)
    for function, args in old_signatures.items():
        if function in new_signatures and args != new_signatures[function]:
            print(f"Signature changed for {function}: {args} -> {new_signatures[function]}")


def compare_imports(old_code: str, new_code: str) -> None:
    old_imports = get_imports(old_code)
    new_imports = get_imports(new_code)
    for old_import in old_imports:
        if old_import not in new_imports:
            print(f"Import removed: {old_import}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python your_script.py <path_to_changed_files.txt>")
        sys.exit(1)

    file_path = sys.argv[1]
    changed_files = get_changed_files(file_path)

    for file_path in changed_files:
        file_path = file_path.strip()  # Remove any leading/trailing whitespace

        if not Path(file_path).exists():
            print(f"File not found: {file_path}. Skipping...")
            continue

        print(f"Analyzing {file_path}...")

        # Get old and new code
        old_code = get_file_content(file_path)
        new_code = get_file_content(file_path)  # Assuming both old and new code are in the same file

        # Compare signatures and imports
        compare_signatures(old_code, new_code)
        compare_imports(old_code, new_code)


if __name__ == "__main__":
    main()
