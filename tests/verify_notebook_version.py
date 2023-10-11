import re
import sys
import super_gradients
import nbformat


def get_first_cell_content(notebook_path):
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


def main():
    """
    This script is used to verify that the version of the SG package matches the version of SG installed in the notebook.
    The script assumes that the first cell of the notebook contains a `!pip install super_gradients=={version}` command.
    :return: An exit code of 0 if the versions match, 1 otherwise.
    """
    notebook_path = sys.argv[1]
    first_cell_content = get_first_cell_content(notebook_path)
    print(first_cell_content)

    # Check if the first cell contains "!pip install super_gradients=={version}" using regex and extract the version
    pattern = re.compile(r"^!pip install super_gradients==([\d\.]+)")

    for line in first_cell_content.splitlines():
        match = re.search(pattern, line)
        if match:
            sg_version_in_notebook = match.group(1)
            if sg_version_in_notebook == super_gradients.__version__:
                return 0
            else:
                print(
                    f"Version mismatch detected:\n"
                    f"super_gradients.__version__ is {super_gradients.__version__}\n"
                    f"Notebook uses super_gradients  {sg_version_in_notebook} (notebook_path)"
                )
                return 1

    print("First code cell of the notebook does not contain a `!pip install super_gradients=={version} command`")
    return 1


if __name__ == "__main__":
    sys.exit(main())
