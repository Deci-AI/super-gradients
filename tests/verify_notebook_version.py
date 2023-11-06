import re
import sys
from typing import Optional

import super_gradients


def get_first_cell_content(notebook_path):
    import nbformat

    # Load the notebook
    with open(notebook_path, "r", encoding="utf-8") as notebook_file:
        notebook_content = nbformat.read(notebook_file, as_version=4)

    # Get the content of the first cell (assuming it's a code cell)
    code_cells = [cell for cell in notebook_content.cells if cell.cell_type == "code"]
    first_cell_content = ""
    if code_cells:
        first_cell = code_cells[0]
        first_cell_content = first_cell.source

    return first_cell_content


def try_extract_super_gradients_version_from_pip_install_command(input: str) -> Optional[str]:
    """
    Extracts the version of super_gradients from a string like `!pip install super_gradients=={version}` command.
    A pip install may contain extra arguments, e.g. `!pip install -q super_gradients=={version} torch=={another version}`.

    :param input: A string that contains a `!pip install super_gradients=={version}` command.
    :return: The version of super_gradients.
    """
    pattern = re.compile(r"pip\s+install.*?super[-_]gradients==([0-9]+(?:\.[0-9]+)*(?:\.[0-9]+)?)")
    match = re.search(pattern, input)
    if match:
        return match.group(1)
    else:
        return None


def main():
    """
    This script is used to verify that the version of the SG package matches the version of SG installed in the notebook.
    The script assumes that the first cell of the notebook contains a `!pip install super_gradients=={version}` command.
    :return: An exit code of 0 if the versions match, 1 otherwise.
    """
    notebook_path = sys.argv[1]
    first_cell_content = get_first_cell_content(notebook_path)

    expected_version = super_gradients.__version__
    for line in first_cell_content.splitlines():
        sg_version_in_notebook = try_extract_super_gradients_version_from_pip_install_command(line)
        if sg_version_in_notebook is not None:
            if sg_version_in_notebook == expected_version:
                return 0
            else:
                print(
                    f"Version mismatch detected in {notebook_path}:\n"
                    f"super_gradients.__version__ is {expected_version}\n"
                    f"Notebook uses super_gradients  {sg_version_in_notebook} (notebook_path)"
                )
                return 1

    print(f"First code cell of the notebook {notebook_path} does not contain a `!pip install super_gradients=={expected_version} command`")
    print("First code cell content:")
    print(first_cell_content)
    return 1


if __name__ == "__main__":
    sys.exit(main())
