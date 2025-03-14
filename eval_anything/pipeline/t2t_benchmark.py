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
from eval_anything.dataloader.t2t_dataloader import T2TDataLoader
from eval_anything.utils.data_type import EvaluationResult
import json
import os
from typing import Dict, List
from eval_anything.utils.data_type import InferenceInput
from eval_anything.utils.utils import pair_data_via_uuid
from collections import namedtuple
from eval_anything.utils.register import BenchmarkRegistry
from eval_anything.utils.logger import EvalLogger

@BenchmarkRegistry.register("text_to_text")
class T2TBenchmark(BaseBenchmark):
    def init_dataloader(self, eval_cfgs: namedtuple, benchmark_cfgs: namedtuple):
        dataset = T2TDataLoader(eval_cfgs, benchmark_cfgs, self.logger)
        return dataset

    def save_benchmark_details(self, save_path: str, benchmark_name: str, inputs: Dict[str, List[InferenceInput]], results: Dict[str, List[EvaluationResult]]):
        """Save evaluation result and config file of single benchmark.
        Args:
            save_path (str): save path
            inputs (dict[str, List[InferenceInput]]): evaluation inputs
            results (dict[str, List[EvaluationResult]]): evaluation results
        """
        output_dir = os.path.join(save_path, benchmark_name)
        os.makedirs(output_dir, exist_ok=True)
        for task, input_data in inputs.items():
            result_data = results[task]
            details = pair_data_via_uuid(input_data, result_data)
            save_details = []
            for input, output in details:
                save_details.append(
                    {
                        "input": input.to_dict(),
                        "output": output.to_dict()
                    }
                )
            with open(os.path.join(output_dir, f"details_{task}.jsonl"), 'a', encoding='utf-8') as f:
                for detail in save_details:
                    f.write(json.dumps(detail, ensure_ascii=False) + '\n')