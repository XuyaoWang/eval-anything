"""
mm dataloader基类
输入：
    - 数据集路径
    - split
    - size
    - 模态
    - 预处理方式（是否pre-tokenize）
    - 模型路径（如果需要pre-tokenize）
    - shuffle
    - num_workers
    - chat_template
    - ...
输出：
    - InferenceInput类
"""

import os
from typing import Any, Dict, List
from collections import namedtuple
from datasets import load_dataset

from eval_anything.dataloader.base_dataloader import BaseDataLoader, TASK_TYPE_MAP
from eval_anything.utils.register import MMDatasetRegistry
from eval_anything.utils.data_type import InferenceInput
from eval_anything.utils.utils import get_project_root

class MMDataLoader(BaseDataLoader):
    
    def load_dataset(self, task_list: list[str]) -> Dict[str, List[InferenceInput]]:
        prompts = {}
        if task_list == []:
            task_list = list(self.bench_cfgs.dataset.default_task_list)
        for task in self.task_info:
            if task.name not in task_list:
                continue
            if task.data_files:
                dataset = load_dataset(self.data_dir, data_files=task.data_files, split="train")
            else:
                dataset = load_dataset(self.data_dir, task.name, split=self.bench_cfgs.dataset.split)
            dataset_formatter = MMDatasetRegistry.get_mm_dataset(self.benchmark_name.lower())(
                bench_cfgs=self.bench_cfgs,
                task=task,
                enable_cot=self.enable_cot,
                num_shot=self.num_shot
            )
            prompts[task.name] = dataset_formatter(dataset)
        return prompts

    def set_fewshot_dataset(self, task: str):
        if self.enable_cot:
            cot_fewshot_data_split = self.bench_cfgs.dataset.cot_fewshot_data_split if self.bench_cfgs.dataset.cot_fewshot_data_split else 'train'
            try:
                data_path = os.path.join(get_project_root(), "eval_anything", "benchmarks", "cot_fewshot", self.bench_cfgs.dataset.cot_fewshot_data_path)
                if os.path.exists(data_path):
                    few_shot_data = load_dataset(data_dir=data_path, data_files=self.bench_cfgs.dataset.cot_fewshot_data_name, split=cot_fewshot_data_split)
                    return few_shot_data
                else:
                    data_name = self.bench_cfgs.dataset.cot_fewshot_data_name.format(task_name=task)
                    few_shot_data = load_dataset(path=self.bench_cfgs.dataset.cot_fewshot_data_path, name=data_name, split=cot_fewshot_data_split)
                    return few_shot_data
            except:
                self.logger.log('error', f"Chain of thought fewshot is not supported for task {self.bench_cfgs.dataset.name}: {task}")
                raise
        else:
            fewshot_data_split = self.bench_cfgs.dataset.fewshot_data_split if self.bench_cfgs.dataset.fewshot_data_split else 'train'
            try:
                data_path = os.path.join(get_project_root(), "eval_anything", "benchmarks", "fewshot", self.bench_cfgs.dataset.fewshot_data_path)
                if os.path.exists(data_path):
                    few_shot_data = load_dataset(data_dir=data_path, data_files=self.bench_cfgs.dataset.fewshot_data_name, split=fewshot_data_split)
                    return few_shot_data
                else:
                    data_name = self.bench_cfgs.dataset.fewshot_data_name.format(task_name=task)
                    few_shot_data = load_dataset(path=self.bench_cfgs.dataset.fewshot_data_path, name=data_name, split=fewshot_data_split)
                    return few_shot_data
            except:
                self.logger.log('error', f"Fewshot is not supported for task {self.bench_cfgs.dataset.name}: {task}")
                raise