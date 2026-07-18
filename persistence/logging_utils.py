import logging
from contextlib import contextmanager

from IPython.utils.capture import capture_output


@contextmanager
def log_jupyter_outputs_to_file(file_path: str, level=logging.INFO):
    file_logger = logging.getLogger("file_logger")
    file_logger.setLevel(level)

    file_handler = logging.FileHandler(file_path, mode="w", encoding="utf-8")
    file_handler.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)

    file_logger.addHandler(file_handler)

    with capture_output() as captured:
        yield

    # Log captured stdout
    if captured.stdout:
        file_logger.log(level, "Captured stdout:\n" + captured.stdout)

    # Log captured stderr
    if captured.stderr:
        file_logger.error("Captured stderr:\n" + captured.stderr)

    # Log display outputs (plots, display(), etc.)
    for idx, output in enumerate(captured.outputs, start=1):
        file_logger.log(level, f"Captured display output {idx}:\n{output}")

    file_logger.removeHandler(file_handler)
    file_handler.close()
