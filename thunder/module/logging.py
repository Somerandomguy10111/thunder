from __future__ import annotations

import traceback
import linecache
import os
from holytools.logging import make_logger
import torch
# ---------------------------------------------------------

thunderLogger = make_logger()

def log_relevant_stacktrace(err: Exception):
    err_class, err_instance, err_traceback = err.__class__, err, err.__traceback__
    tb_list = traceback.extract_tb(err_traceback)

    def is_relevant(tb):
        not_torch = not os.path.dirname(torch.__file__) in tb.filename
        return not_torch

    relevant_tb = [tb for tb in tb_list if is_relevant(tb)]

    if relevant_tb:
        err_msg = "\nEncountered error during training routine. Relevant stacktrace:"
        for frame in relevant_tb:
            file_path = frame.filename
            line_number = frame.lineno
            tb_str = (f'File "{file_path}", line {line_number}, in {frame.name}\n'
                      f'    {linecache.getline(file_path, line_number).strip()}')
            err_msg += f'\n{err_class.__name__}: {err_instance}\n{tb_str}'
        thunderLogger.critical(msg=err_msg)

