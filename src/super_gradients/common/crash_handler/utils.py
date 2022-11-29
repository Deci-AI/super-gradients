import json
from termcolor import colored


def indent_string(txt: str, indent_size: int) -> str:
    """Add an indentation to a string."""
    indent = " " * indent_size
    return indent + txt.replace("\n", "\n" + indent)


def fmt_txt(txt: str, bold: bool = False, color: str = "", indent: int = 0) -> str:
    """Format a text for the console."""
    if bold:
        BOLD = "\033[1m"
        END = "\033[0m"
        txt = BOLD + txt + END
    if color:
        txt = colored(txt, color)
    if indent:
        txt = indent_string(txt, indent_size=indent)
    return txt


def json_str_to_dict(json_str: str) -> dict:
    """Build a dictionary from a string in some sort of format."""
    json_str = json_str.replace("None", '"None"').replace("'", '"')
    return json.loads(json_str)
