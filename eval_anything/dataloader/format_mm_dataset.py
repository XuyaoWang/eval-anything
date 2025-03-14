from typing import List, Dict, Union
import os
import re
import ast
import base64
import PIL
from PIL import Image
from io import BytesIO
from eval_anything.utils.register import MMDatasetRegistry
from eval_anything.utils.data_type import InferenceInput
import eval_anything.utils.utils as utils
from eval_anything.utils.utils import MultiChoicePromptBuilder, DialoguePromptBuilder
from eval_anything.dataloader.base_dataloader import TASK_TYPE_MAP
from datasets import Dataset, load_dataset
from collections import namedtuple

class BaseMMDataset:
    def __init__(self, bench_cfgs: namedtuple, task: namedtuple, enable_cot: bool, num_shot: int):
        self.bench_cfgs = bench_cfgs
        self.task = task
        self.enable_cot = enable_cot
        self.num_shot = num_shot
        self.few_shot_examples = []
        self.few_shot_mm_examples = []

    def set_few_shot_examples(self, few_shot_dataset: Dataset | None):
        for item in few_shot_dataset[:self.num_shot]:
            self.few_shot_examples.append({
                "question": item[self.task.question_key],
                "candidate_answers": item[self.task.answer_key],
                "ground_truth": item[self.task.ground_truth_key]
            })        

    def build_multi_choice_prompt(self, item: dict):
        self.prompt_builder = MultiChoicePromptBuilder(
            candidate_labels=self.task.candidate_labels,
            few_shot_examples=self.few_shot_examples,
            cot=self.enable_cot
        )
        prompt = self.prompt_builder.build_prompt(item[self.task.question_key], item[self.task.answer_key])
        return prompt

    def build_dialogue_prompt(self, item: dict):
        self.prompt_builder = DialoguePromptBuilder(
            few_shot_examples=self.few_shot_examples,
            cot=self.enable_cot
        )
        prompt = self.prompt_builder.build_prompt(item[self.task.question_key], item[self.task.answer_key])
        return prompt
    
    def encode_image_to_base64(self, image: Union[str, "PIL.Image"]) -> str:
        """Get base64 from image"""
        if isinstance(image, str):
            image_input = Image.open(image)
        else:
            image_input = image
        
        if image_input.mode != "RGB":
            image_input = image_input.convert("RGB")

        buffer = BytesIO()
        image_input.save(buffer, format="JPEG")
        img_bytes = buffer.getvalue()
        base64_data = base64.b64encode(img_bytes).decode("utf-8")
        return base64_data
    
    def _to_InferenceInput(self, dataset: Dataset):
        pass

    def __call__(self, dataset: Dataset):
        return self._to_InferenceInput(dataset)

@MMDatasetRegistry.register("mmmu")
class MMMUDataset(BaseMMDataset):
    def __init__(self, bench_cfgs: namedtuple, task: namedtuple, enable_cot: bool, num_shot: int):
        super().__init__(bench_cfgs, task, enable_cot, num_shot)

    def get_image_indice(self, text: str)->List[int]:
        pattern = r'<image (\d+)>'
        matches = re.findall(pattern, text)
        return [int(num) for num in matches]
    
    def prompt_to_conversation(
            self, 
            user_prompt: str, 
            system_prompt: Union[str, None] = None, 
            images: Union[List[PIL.Image], List[str]] = []
        ) -> List[Dict]:
        """
        Convert input prompt to the specified conversation format
        
        Args:
            user_prompt (str): Input user_prompt with image placeholders in <image n> format
            system_prompt (str): Input system_prompt (if exists)
            images (list): List of PIL.Image objects to be encoded and inserted into the conversation
            
        Returns:
            list: Conversation object in the specified format
        """
        image_pattern = re.compile(r'<image (\d+)>')
        matches = list(image_pattern.finditer(user_prompt))
        assert len(images) == len(matches), f"Number of images ({len(images)}) does not match number of placeholders ({len(matches)}), input user_prompt: {user_prompt}"
        
        content_parts = []
        
        if not matches:
            if user_prompt:
                content_parts.append({
                    "type": "text",
                    "text": user_prompt
                })
        else:
            if matches[0].start() > 0:
                content_parts.append({
                    "type": "text",
                    "text": user_prompt[:matches[0].start()]
                })
            
            for i, match in enumerate(matches):
                content_parts.append({
                    "type": "image",
                    "image": f"data:image/jpeg;base64,{self.encode_image_to_base64(images[i])}"
                })
                
                text_start = match.end()
                text_end = matches[i+1].start() if i+1 < len(matches) else len(user_prompt)
                
                if text_end > text_start:
                    content_parts.append({
                        "type": "text",
                        "text": user_prompt[text_start:text_end]
                    })

        conversation = [
            {
                "role": "user",
                "content": content_parts
            }
        ]

        if system_prompt is not None:
            conversation.insert(0, {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_prompt
                    }
                ]
            })
        
        return conversation

    def set_few_shot_examples(self, few_shot_dataset: Dataset | None):
        raise NotImplementedError("MMMU does not support few-shot learning.")

    # refer: https://github.com/MMMU-Benchmark/MMMU/blob/main/mmmu/utils/data_utils.py#L136
    def _to_InferenceInput(self, dataset: Dataset) -> List["InferenceInput"]:
        """
        Convert a dataset to a list of InferenceInput objects.
        
        Args:
            dataset: Dataset object containing questions, options, and images
            
        Returns:
            List of InferenceInput objects ready for model inference
        """
        inference_inputs = []
        
        for item in dataset:
            question = item['question']
            if item['question_type'] == 'multiple-choice':
                options = eval(item['options'])
                example = ""
                letter_to_option = {}
                
                for idx, option in enumerate(options):
                    option_letter = chr(ord('A') + idx)
                    example += f"({option_letter}) {option}\n"
                    letter_to_option[option_letter] = option
                
                formatted_prompt = f"{question}\n\n{example}\n\nAnswer with the option's letter from the given choices directly."
            else:
                formatted_prompt = f"{question}\n\nAnswer the question using a single word or phrase."
            
            image_ids = self.get_image_indice(formatted_prompt)
            images = [item[f'image_{id}'] for id in image_ids]
            conversation = self.prompt_to_conversation(formatted_prompt, images=images)

            inference_inputs.append(
                InferenceInput(
                    task=self.task.name,
                    conversation=conversation,
                    ref_answer=item['answer']
                )
            )
        return inference_inputs