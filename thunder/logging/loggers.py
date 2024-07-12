from __future__ import annotations

import os
import tempfile

from GPUtil import GPU
from git import Repo, InvalidGitRepositoryError
from wandb.sdk.wandb_run import Run

from holytools.logging import LoggerFactory
from .metric import Metric


# ---------------------------------------------------------
class WBLogger:
    def __init__(self, run : Run):
        self.run : Run = run
        self.current_batch : int = 0
        self.current_epoch : int = 0
        self.completed_subruns : int = 0
        self.console_logger = LoggerFactory.make_logger(name=self.__class__.__name__)

    @classmethod
    def wandb_is_available(cls) -> bool:
        if os.getenv('WANDB_API_KEY'):
            return True
        elif os.path.isfile(os.path.expanduser('~/.netrc')):
            return True
        else:
            return False

    def finish_subrun(self):
        self.log_code_state()
        self.current_epoch = 0
        self.current_batch = 0
        self.completed_subruns += 1

    # ---------------------------------------------------------
    # increment

    def increment_epoch(self):
        self.current_epoch += 1

    def increment_batch(self):
        self.current_batch += 1

    def log_code_state(self):
        repo_path = os.getcwd()
        try:
            repo = Repo(repo_path, search_parent_directories=True)
            commit_hash = repo.head.commit.hexsha
            git_diff = repo.git.diff()
            with tempfile.NamedTemporaryFile(delete=False, mode='w+', suffix='.diff') as tmpfile:
                content = f'Git diff for commit {commit_hash}' + '\n' + 50*'-' + '\n'
                content += git_diff

                tmpfile.write(content)
                tmpfile_path = tmpfile.name
            self.run.log_artifact(artifact_or_path=tmpfile_path,name=f'git_diff',type='code')

            self.console_logger.info(f"Logged current code state as commit {commit_hash} and diff from {tmpfile_path}")
        except InvalidGitRepositoryError:
            self.console_logger.warning(f"Failed to log code state because {repo_path} is not a Git repository.")
        except Exception as e:
            self.console_logger.warning(f"Failed to log code state with error: {e}")

    # ---------------------------------------------------------
    # logging (interface)

    def log_metrics(self, metric_map : dict[str, Metric], is_training: bool):
        for k, v in metric_map.items():
            if is_training:
                self.log_training_quantity(name=k, value=v.value)
            else:
                self.log_validation_quantity(name=k, value=v.value)

    def log_gpu_resources(self, gpus : list[GPU]):
        for gpu in gpus:
            gpu_memory_load_factor = (gpu.memoryTotal - gpu.memoryFree) / gpu.memoryTotal
            self.log_system(name=f'GPU {gpu.id} free memory in GB', value=gpu.memoryFree / 1024)
            self.log_system(name=f'GPU {gpu.id} memory load', value=gpu_memory_load_factor)
        free_gpu_memory_mb = sum([gpu.memoryFree for gpu in gpus])
        total_gpu_memory_mb = sum([gpu.memoryTotal for gpu in gpus])
        memory_load_factor = (total_gpu_memory_mb - free_gpu_memory_mb) / total_gpu_memory_mb
        self.log_system(name='Free GPU memory in GB', value=free_gpu_memory_mb / 1024)
        self.log_system(name='GPU memory load', value=memory_load_factor)

    def log_validation_quantity(self, name: str, value: float):
        self.log_quantity(name=f'Validation/{name}', value=value)

    def log_training_quantity(self, name: str, value: float):
        self.log_quantity(name=f'Training/{name}', value=value)

    def log_system(self, name : str, value : float):
        self.log_quantity(name=f'System/{name}', value=value)

    def log_quantity(self, name: str, value: float):
        conditional_subruns = f'_{self.completed_subruns}' if self.completed_subruns > 0 else ''
        self._log(metric_dict={f'{name}{conditional_subruns}': value})

    # ---------------------------------------------------------
    # logging (internal)

    def _log(self, metric_dict: dict[str, int | float]):
        metric_dict['epoch'] = self.current_epoch
        metric_dict['batch'] = self.current_batch
        metric_dict['subruns'] = self.completed_subruns
        self.run.log(data=metric_dict)




