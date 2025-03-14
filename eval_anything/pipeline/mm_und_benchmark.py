"""
t2t任务基类，不直接使用，而是继承后实现具体任务的逻辑
输入：
    - 数据集路径
    - 模型路径
    - 模态
    - 预处理方式（是否pre-tokenize）
    - 模型推理方式(调用eval-anything/models中的推理方式)
    - ...
输出：
    - EvaluationResult类
"""

from eval_anything.pipeline.base_benchmark import BaseBenchmark
from eval_anything.dataloader.mm_dataloader import MMDataLoader
from eval_anything.utils.data_type import EvaluationResult
import json
import os
from typing import Dict, List
from eval_anything.utils.data_type import InferenceInput
from eval_anything.utils.utils import pair_data_via_uuid
from collections import namedtuple
from eval_anything.utils.register import BenchmarkRegistry, MMDatasetRegistry
from eval_anything.utils.logger import EvalLogger

@BenchmarkRegistry.register("mm_und")
class MMUndBenchmark(BaseBenchmark):
    def init_dataloader(self, eval_cfgs: namedtuple, benchmark_cfgs: namedtuple):
        dataset = MMDataLoader(eval_cfgs, benchmark_cfgs, self.logger)
        return dataset

    # TODO
    def save_benchmark_details(self, save_path: str, benchmark_name: str, inputs: Dict[str, List[InferenceInput]], results: Dict[str, List[EvaluationResult]]):
        """Save evaluation result and config file of single benchmark.
        Args:
            save_path (str): save path
            inputs (dict[str, List[InferenceInput]]): evaluation inputs
            results (dict[str, List[EvaluationResult]]): evaluation results
        """
        pass