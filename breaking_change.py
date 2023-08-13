import subprocess
import ast
import sys
from typing import List, Dict


def get_changed_files(file_path: str) -> List[str]:
    with open(file_path, "r") as file:
        return [line.strip() for line in file]


def get_file_content(file_path: str, branch: str = None) -> str:
    if branch:
        result = subprocess.run(["git", "show", f"{branch}:{file_path}"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            print(f"Error getting content from {branch} for {file_path}: {result.stderr.decode()}")
            return ""
        return result.stdout.decode()
    else:
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


def compare_signatures(old_code: str, new_code: str):
    old_sigs = get_function_signatures(old_code)
    new_sigs = get_function_signatures(new_code)

    report = []
    for name, old_sig in old_sigs.items():
        new_sig = new_sigs.get(name)
        if new_sig and old_sig != new_sig:
            details = f"Signature changed for {name}: {old_sig} -> {new_sig}"
            report.append(details)
    return report


def compare_imports(old_code: str, new_code: str):
    old_imports = get_imports(old_code)
    new_imports = get_imports(new_code)

    report = []
    for old_import in old_imports:
        if old_import not in new_imports:
            details = f"Import removed: {old_import}"
            report.append(details)
    return report


def main():
    if len(sys.argv) != 2:
        print("Usage: python your_script.py <path_to_changed_files.txt>")
        sys.exit(1)

    file_path = sys.argv[1]
    changed_files = get_changed_files(file_path)

    for file_path in changed_files:
        file_path = file_path.strip()  # Remove any leading/trailing whitespace

        print(f"Analyzing {file_path}...")

        # Get old and new code
        old_code = get_file_content(file_path, "master")
        new_code = get_file_content(file_path)

        signature_changes = compare_signatures(old_code, new_code)
        import_removed = compare_imports(old_code, new_code)

    with open("report.txt", "w") as file:
        for details in signature_changes + import_removed:
            file.write(details + "\n\n")
            print(details)


if __name__ == "__main__":
    main()
