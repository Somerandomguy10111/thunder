import traceback
import linecache
import pytorch_lightning
import os
from holytools.logging import make_logger
import torch
from .configs import WBConfig, RunConfigs, Logger
# ---------------------------------------------------------

thunderLogger = make_logger()

def log_relevant_stacktrace(err: Exception):
    err_class, err_instance, err_traceback = err.__class__, err, err.__traceback__
    tb_list = traceback.extract_tb(err_traceback)

    def is_relevant(tb):
        not_lightning = not os.path.dirname(pytorch_lightning.__file__) in tb.filename
        not_torch = not os.path.dirname(torch.__file__) in tb.filename
        return not_lightning and not_torch

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


def get_wb_logger(run_configs: RunConfigs) -> Logger:
    kwargs = {
        "lr": run_configs.descent.lr,
        "batch_size": run_configs.batch_size,
        "epochs": run_configs.epochs,
        "optimizer": run_configs.descent.get_algorithm().__name__,
        "seed": run_configs.seed
    }
    logger = WBConfig(**kwargs).get_logger()
    if not os.path.isdir(logger.save_dir):
        os.makedirs(logger.save_dir)
    return logger
