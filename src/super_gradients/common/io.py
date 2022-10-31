import sys
from datetime import datetime

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.environment.environment_config import PKG_CHECKPOINTS_DIR
from super_gradients.common.environment.env_helpers import multi_process_safe, is_distributed


logger = get_logger(__name__)
CONSOLE_LOG_PATH = f"{PKG_CHECKPOINTS_DIR}/console.log"  # TODO: move to experiment, but how do we get it ?


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


@multi_process_safe
def log_std_streams():
    """Log the standard streams (stdout/stderr) into a local file."""

    f = open(CONSOLE_LOG_PATH, "a")
    if not is_distributed():
        f.write("\n\n============================================================\n")
        f.write(f'New run started at {datetime.today().isoformat()}\n')
        f.write(f'sys.argv: "{" ".join(sys.argv)}"\n')
        f.write("============================================================\n")

    StdoutTee(f)
    StderrTee(f)

    logger.info(f"The console stream is logged into {CONSOLE_LOG_PATH}")
