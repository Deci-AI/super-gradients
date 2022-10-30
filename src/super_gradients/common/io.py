import sys
import logging
from datetime import datetime
from pathlib import Path

from super_gradients.common.abstractions.abstract_logger import get_logger

logger = get_logger(__name__, log_level=logging.DEBUG)


class StdoutTee(object):
    def __init__(self, file):
        self.file = file
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()


class StderrTee(object):
    def __init__(self, file):
        self.file = file
        self.stderr = sys.stderr
        sys.stderr = self

    def __del__(self):
        sys.stderr = self.stderr
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stderr.write(data)

    def flush(self):
        self.file.flush()


def log_std_streams():
    """Log the standard streams (stdout/stderr) into a local file."""
    current_time = datetime.today().isoformat()

    # TODO: check how we handle the log file names, I dont think this is the right way to do
    file_path = Path(__file__)  # super-gradients/src/super_gradients/sanity_check/env_sanity_check.py
    package_root = file_path.parent.parent  # super-gradients
    log_file = package_root / f"console.log.{current_time}"

    f = open(log_file, "a")
    f.write(f'Run from {current_time}\n')
    f.write(f'sys.argv: "{" ".join(sys.argv)}"\n')
    f.write('-' * 20 + "\n\n")

    StdoutTee(f)
    StderrTee(f)

    logger.info(f"The console stream is logged into {log_file}")
