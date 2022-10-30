import sys
from datetime import datetime
import logging

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


def _log_std_streams():
    """Log the standard streams (stdout/stderr) into a local file."""
    current_time = datetime.today().isoformat()
    log_file = f"/home/louis.dupont/PycharmProjects/super-gradients/console.log.{current_time}"

    f = open(log_file, "a")
    f.write(f'Run from {current_time}\n')
    f.write(f'sys.argv: "{" ".join(sys.argv)}"\n')
    f.write('-' * 20 + "\n\n")

    StdoutTee(f)
    StderrTee(f)

    logger.info(f"The console stream is logged into {log_file}")


# This is called on import.
# TODO: Do we want to call it by default or to only call for pro users ?
# TODO: Should this be the default behavior, or we only use this when LOG_STD_STREAMS=True ?
_log_std_streams()
