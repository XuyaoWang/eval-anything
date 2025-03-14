"""
任务基类，不直接使用，而是继承后实现具体任务的逻辑

TODO 
    - 代码鲁棒性
    - logger
"""
from abc import ABC, abstractmethod
import os
import hashlib
import importlib
import json
import yaml
from typing import Dict, List
from collections import namedtuple

# Third-party imports
# from vllm.sequence import Logprob
# from vllm.sequence import RequestOutput

# Local imports
from eval_anything.utils.logger import EvalLogger
from eval_anything.models.base_model import MODEL_MAP, CLASS_MAP
from eval_anything.evaluate_tools.t2t_tools import *
import eval_anything.evaluate_tools.t2t_tools as T2T_TOOLS
from eval_anything.utils.utils import (
    read_cfgs_from_yaml, update_dict, 
    custom_cfgs_to_dict, BENCHMARK_MODALITY_MAP, pair_data_via_uuid,
    dict_to_namedtuple, namedtuple_to_dict
)
from eval_anything.utils.cache_manager import CacheManager
from eval_anything.utils.register import BenchmarkRegistry
import eval_anything.benchmarks
import eval_anything.dataloader.format_mm_dataset



class BaseTask(ABC):
    def __init__(self, overall_cfgs_name: str, **kwargs):
        self.eval_cfgs, self.model_cfgs, self.infer_cfgs = self.get_overall_configs(overall_cfgs_name, **kwargs)
        self.enable_cache = True if self.eval_cfgs.cache_dir else False
        self.output_path = self.eval_cfgs.output_dir
        self.logger = EvalLogger('Eval-Anything', log_dir=self.output_path)
        self.cache_manager = CacheManager(self.eval_cfgs.cache_dir, self.logger) if self.enable_cache else None
        self.model = self.load_model(self.model_cfgs, self.infer_cfgs)

    def get_overall_configs(self, overall_cfgs_name: str, **kwargs):
        """Get configs from yaml file
        Args:
            overall_cfgs_name (str): YAML filename for evaluation configurations in eval-anything/configs/.
            
        Returns:
            eval_cfgs (dict): eval configs, including
            model_cfgs (dict): model configs
            infer_cfgs (dict): infer configs
        """
        overall_cfgs = read_cfgs_from_yaml(yaml_relative_dir='configs', yaml_name=overall_cfgs_name)
        eval_cfgs = overall_cfgs['eval_cfgs']
        model_cfgs = overall_cfgs['model_cfgs']
        infer_cfgs = overall_cfgs['infer_cfgs']
        for k, v in kwargs.items():
            if v == '' or v is None:
                continue
            eval_cfgs = update_dict(eval_cfgs, custom_cfgs_to_dict(k, v))
            model_cfgs = update_dict(model_cfgs, custom_cfgs_to_dict(k, v))
            infer_cfgs = update_dict(infer_cfgs, custom_cfgs_to_dict(k, v))
        eval_cfgs = dict_to_namedtuple(eval_cfgs)
        model_cfgs = dict_to_namedtuple(model_cfgs)
        infer_cfgs = dict_to_namedtuple(infer_cfgs)
        return eval_cfgs, model_cfgs, infer_cfgs
    
    def get_benchmark_dict(self) -> dict[str, list[str]]:
        """Get benchmark name from self.task_cfgs
            
        Returns:
            benchmark_dict (dict[str, list[str]]): {benchmark_name: [task_name1, task_name2, ...]}
        """
        return self.eval_cfgs.benchmarks._asdict()

    def load_model(self, model_cfgs: namedtuple, infer_cfgs: namedtuple):
        backend_type = f"{infer_cfgs.infer_backend}_{model_cfgs.model_type}"
        module_name = f"eval_anything.models.{MODEL_MAP[backend_type]}"
        module = importlib.import_module(module_name)
        model_class = getattr(module, CLASS_MAP[backend_type])
        model = model_class(model_cfgs, infer_cfgs)
        return model
    
    def iterate_run(self) -> list[dict[str, dict[str, float]]]:
        """Iterate benchmark list and run benchmarks"""
        self.results = {}
        self.benchmark_dict = self.get_benchmark_dict()
        overall_results = {}
        for benchmark_name, task_list in self.benchmark_dict.items():
            benchmark_evaluator = BenchmarkRegistry.get_benchmark(benchmark_name)(self.model, self.eval_cfgs, self.model_cfgs, self.infer_cfgs, self.output_path, self.cache_manager, self.logger)
            evaluation_details, evaluation_results, overall_result = benchmark_evaluator.run(task_list)
            self.results[benchmark_name] = evaluation_results
            overall_results[benchmark_name] = overall_result
        self.display_task_results(overall_results)
        self.save_task_details(self.output_path)
        return self.results
    
    def shutdown_model(self):
        """Shutdown model"""
        self.model.shutdown()

    def display_task_results(self, results: dict[str, dict[str, dict[str, float]]]):
        """Display overall evaluation results in command line.
        Args:
            results (list[dict[str, dict[str, float]]]): evaluation results
        """
        for benchmark_name, result in results.items():
            if result != {}:
                self.logger.print_table(title=f'Brief Report of {benchmark_name}', data=result, to_csv=True, csv_file_name=f'{benchmark_name}_brief_report.csv')
    
    def save_task_details(self, save_path: str):
        """Save overall evaluation results, including all benchmark results.
        Args:
            save_path (str): save path
            results (list[EvaluationResult]): evaluation results
        """
        output_dict = {
            'eval_cfgs': namedtuple_to_dict(self.eval_cfgs),
            'model_cfgs': namedtuple_to_dict(self.model_cfgs),
            'infer_cfgs': namedtuple_to_dict(self.infer_cfgs),
            'results': self.results
        }
        with open(os.path.join(save_path, 'evaluation_details.json'), 'w') as f:
            json.dump(output_dict, f, indent=4)