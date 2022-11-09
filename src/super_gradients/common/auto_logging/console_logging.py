# import os
# import sys
# from datetime import datetime
# from pathlib import Path
# from super_gradients.common.environment.env_helpers import multi_process_safe, is_distributed
#
#
# class StdoutTee(object):
#     """Duplicate the stdout stream to save it into a given file."""
#
#     def __init__(self, file):
#         self.file = file
#         self.stdout = sys.stdout
#         sys.stdout = self
#
#     def __del__(self):
#         sys.stdout = self.stdout
#         self.file.close()
#
#     def write(self, data):
#         self.file.write(data)
#         self.stdout.write(data)
#
#     def flush(self):
#         self.file.flush()
#
#     def __getattr__(self, attr):
#         return getattr(self.stdout, attr)
#
#
# class StderrTee(object):
#     """Duplicate the stderr stream to save it into a given file."""
#
#     def __init__(self, file):
#         self.file = file
#         self.stderr = sys.stderr
#         sys.stderr = self
#
#     def __del__(self):
#         sys.stderr = self.stderr
#         self.file.close()
#
#     def write(self, data):
#         self.file.write(data)
#         self.stderr.write(data)
#
#     def flush(self):
#         self.file.flush()
#
#     def __getattr__(self, attr):
#         return getattr(self.stderr, attr)
#
#
# def copy_file(src_filename: str, dest_filename: str, copy_mode: str = "w"):
#     """Copy a file from source to destination. Also works when the destination folder does not exist."""
#     os.makedirs(os.path.dirname(dest_filename), exist_ok=True)
#     if os.path.exists(src_filename):
#         with open(src_filename, "r", encoding="utf-8") as src:
#             with open(dest_filename, copy_mode) as dst:
#                 dst.write(src.read())
#
#
# class ConsoleSink:
#     """Singleton responsible to sink the console streams (stdout/stderr) into a file."""
#
#     def __init__(self):
#         self._setup()
#
#     @multi_process_safe
#     def _setup(self):
#         """On instantiation, setup the default sink file."""
#         default_path = Path.home() / "sg_logs" / "console.log"
#         default_path.parent.mkdir(exist_ok=True)
#
#         # By default overwrite existing log. If is_distributed() (i.e. DDP - node 0), append.
#         self.file = open(default_path, mode="w" if not is_distributed() else "a")
#         self.stdout = StdoutTee(self.file)
#         self.stderr = StderrTee(self.file)
#
#         # We don't want to rewrite this for subprocesses when using DDP.
#         if not is_distributed():
#             self.file.write("============================================================\n")
#             self.file.write(f'New run started at {datetime.now().strftime("%Y-%m-%d.%H:%M:%S.%f")}\n')
#             self.file.write(f'sys.argv: "{" ".join(sys.argv)}"\n')
#             self.stdout.write(f"The console stream is logged into {default_path}\n")  # Print this only
#             self.file.write("============================================================\n")
#
#     @multi_process_safe
#     def _set_location(self, filename: str):
#         """Copy and redirect the sink file into another location."""
#
#         prev_file = self.file
#         copy_file(src_filename=prev_file.name, dest_filename=filename, copy_mode="a")
#
#         self.file = open(filename, "a")
#         self.stdout.file = self.file
#         self.stderr.file = self.file
#
#         prev_file.close()
#         self.stdout.write(f"The console stream is now moved to {filename}\n")
#
#     @staticmethod
#     def set_location(filename: str) -> None:
#         """Copy and redirect the sink file into another location."""
#         _console_sink._set_location(filename)
#
#     @staticmethod
#     def get_filename():
#         return _console_sink.file.name
#
#
# _console_sink = ConsoleSink()


import os
import sys
from datetime import datetime
from pathlib import Path
from io import StringIO

from super_gradients.common.environment.env_helpers import multi_process_safe, is_distributed


FILE_BUFFER_SIZE = 10_000


class BufferWriter:
    """Duplicate the stderr stream to save it into a given file."""

    def __init__(self, filename: str, buffer_size: int):
        self.buffer = StringIO()
        self.filename = filename
        self.buffer_size = buffer_size

    def __del__(self):
        self.flush(force=True)

    def write(self, data: str):
        self.buffer.write(data)

    def flush(self, force: bool = False):
        """Flush if the buffer is big enough, or if flush is forced"""
        if force or len(self.buffer.getvalue()) > self.buffer_size:
            with open(self.filename, "a") as f:
                f.write(self.buffer.getvalue())
                self.buffer.truncate(0)


class StderrTee(BufferWriter):
    """Duplicate the stderr stream to save it into a given file."""

    def __init__(self, filename: str, buffer_size: int):
        super().__init__(filename, buffer_size)
        self.stderr = sys.stderr
        sys.stderr = self

    def __del__(self):
        super().__del__()
        self.flush(force=True)
        sys.stderr = self.stderr

    def write(self, data):
        super().write(data)
        self.stderr.write(data)

    def __getattr__(self, attr):
        return getattr(self.stderr, attr)


class StdoutTee(BufferWriter):
    """Duplicate the stdout stream to save it into a given file."""

    def __init__(self, filename: str, buffer_size: int):
        super().__init__(filename, buffer_size)
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        super().__del__()
        self.flush(force=True)
        sys.stdout = self.stdout

    def write(self, data):
        super().write(data)
        self.stdout.write(data)

    def __getattr__(self, attr):
        return getattr(self.stdout, attr)


def copy_file(src_filename: str, dest_filename: str, copy_mode: str = "w"):
    """Copy a file from source to destination. Also works when the destination folder does not exist."""
    os.makedirs(os.path.dirname(dest_filename), exist_ok=True)
    if os.path.exists(src_filename):
        with open(src_filename, "r", encoding="utf-8") as src:
            with open(dest_filename, copy_mode) as dst:
                dst.write(src.read())


class ConsoleSink:
    """Singleton responsible to sink the console streams (stdout/stderr) into a file."""

    def __init__(self):
        self._setup()

    @multi_process_safe
    def _setup(self):
        """On instantiation, setup the default sink file."""
        filename = Path.home() / "sg_logs" / "console.log"
        filename.parent.mkdir(exist_ok=True)
        self.filename = str(filename)

        # We don't want to rewrite this for subprocesses when using DDP.
        if not is_distributed():
            # By default overwrite existing log. If is_distributed() (i.e. DDP - node 0), append.
            with open(self.filename, mode="w" if not is_distributed() else "a") as f:
                f.write("============================================================\n")
                f.write(f'New run started at {datetime.now().strftime("%Y-%m-%d.%H:%M:%S.%f")}\n')
                f.write(f'sys.argv: "{" ".join(sys.argv)}"\n')
                f.write("============================================================\n")

        self.stdout = StdoutTee(filename=self.filename, buffer_size=FILE_BUFFER_SIZE)
        self.stderr = StderrTee(filename=self.filename, buffer_size=FILE_BUFFER_SIZE)
        self.stdout.write(f"The console stream is logged into {self.filename}\n")

    @multi_process_safe
    def _set_location(self, filename: str):
        """Copy and redirect the sink file into another location."""

        prev_filename = self.filename
        copy_file(src_filename=prev_filename, dest_filename=filename, copy_mode="a")

        self.filename = filename
        self.stdout.filename = filename
        self.stderr.filename = filename
        self.stdout.write(f"The console stream is now moved to {filename}\n")

    @staticmethod
    def set_location(filename: str) -> None:
        """Copy and redirect the sink file into another location."""
        _console_sink._set_location(filename)

    @staticmethod
    def get_filename():
        return _console_sink.filename


_console_sink = ConsoleSink()
