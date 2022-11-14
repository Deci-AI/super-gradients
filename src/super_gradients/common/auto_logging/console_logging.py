import os
import sys
from datetime import datetime
from pathlib import Path
from io import StringIO
import atexit
from threading import Lock

from super_gradients.common.environment.env_helpers import multi_process_safe, is_distributed


class BufferWriter:
    """File writer buffer that opens a file only when flushing and under the condition that threshold buffersize was reached."""

    FILE_BUFFER_SIZE = 10_000  # Number of chars to be buffered before writing the buffer on disk.

    def __init__(self, filename: str, buffer_size: int):
        self.buffer = StringIO()
        self.filename = filename
        self.buffer_size = buffer_size
        self.lock = Lock()

    def write(self, data: str):
        """Write to buffer (not on disk)."""
        with self.lock:
            self.buffer.write(data)
        if self._require_flush():
            self.flush()

    def flush(self, force: bool = False):
        """Write the buffer on disk if relevant."""
        if force or self._require_flush():
            with self.lock:
                os.makedirs(os.path.dirname(self.filename), exist_ok=True)
                with open(self.filename, "a", encoding="utf-8") as f:
                    f.write(self.buffer.getvalue())
                    self.buffer = StringIO()

    def _require_flush(self) -> bool:
        """Indicate if a buffer is needed (i.e. if buffer size above threshold)"""
        return len(self.buffer.getvalue()) > self.buffer_size


class StderrTee(BufferWriter):
    """Duplicate the stderr stream to save it into a given file."""

    def __init__(self, filename: str, buffer_size: int):
        super().__init__(filename, buffer_size)
        self.stderr = sys.stderr
        sys.stderr = self

    def __del__(self):
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
            with open(dest_filename, copy_mode, encoding="utf-8") as dst:
                dst.write(src.read())


class ConsoleSink:
    """Singleton responsible to sink the console streams (stdout/stderr) into a file."""

    def __init__(self):
        self._setup()
        atexit.register(self._flush)  # Flush at the end of the process

    @multi_process_safe
    def _setup(self):
        """On instantiation, setup the default sink file."""
        filename = Path.home() / "sg_logs" / "console.log"
        filename.parent.mkdir(exist_ok=True)
        self.filename = str(filename)
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)

        self.stdout = StdoutTee(filename=self.filename, buffer_size=BufferWriter.FILE_BUFFER_SIZE)
        self.stderr = StderrTee(filename=self.filename, buffer_size=BufferWriter.FILE_BUFFER_SIZE)

        # We don't want to rewrite this for subprocesses when using DDP.
        if not is_distributed():
            # By default overwrite existing log. If is_distributed() (i.e. DDP - node 0), append.
            with open(self.filename, mode="w", encoding="utf-8") as f:
                f.write("============================================================\n")
                f.write(f'New run started at {datetime.now().strftime("%Y-%m-%d.%H:%M:%S.%f")}\n')
                f.write(f'sys.argv: "{" ".join(sys.argv)}"\n')
                f.write("============================================================\n")
        self.stdout.write(f"The console stream is logged into {self.filename}\n")

    @multi_process_safe
    def _set_location(self, filename: str):
        """Copy and redirect the sink file into another location."""
        self._flush()

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

    @multi_process_safe
    def _flush(self):
        """Force the flush on stdout and stderr."""
        self.stdout.flush(force=True)
        self.stderr.flush(force=True)

    @staticmethod
    def flush():
        """Force the flush on stdout and stderr."""
        _console_sink._flush()

    @staticmethod
    def get_filename():
        """Get the filename of the sink."""
        return _console_sink.filename


_console_sink = ConsoleSink()
