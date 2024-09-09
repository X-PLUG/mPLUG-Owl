from collections import defaultdict
import copy
import json
import os
from pathlib import Path
import re
import time
from typing import Dict, List
from .base import EvalTaskBase, TASK, ResulterSaverBase, StandardData, load_jsonl
import random
from x.io import read_json, write_json
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import numpy as np
from datasets import load_dataset

multiple_choices = ['A', 'B', 'C', 'D', 'E']

ds_collections = {
    'mmbench_dev_cn_20231003': {
        'raw_file': 'path/to/mmbench_dev_cn_20231003.tsv',
        'annotation': 'path/to/mmbench_dev_cn_20231003.jsonl',
        'image_path': 'images/',
        'only_submit': False
    },
    'mmbench_dev_en_20231003': {
        'raw_file': 'path/to/mmbench_dev_en_20231003.tsv',
        'annotation': 'path/to/mmbench_dev_en_20231003.jsonl',
        'image_path': 'images/',
        'only_submit': False
    },
    'mmbench_test_cn_20231003': {
        'raw_file': 'dataset/mmbench/mmbench_test_cn_20231003.tsv',
        'annotation': 'dataset/mmbench/mmbench_test_cn_20231003.jsonl',
        'image_path': 'images/',
        'only_submit': True
    },
    'mmbench_test_en_20231003': {
        'raw_file': 'dataset/mmbench/mmbench_test_en_20231003.tsv',
        'annotation': 'dataset/mmbench/mmbench_test_en_20231003.jsonl',
        'image_path': 'images/',
        'only_submit': True
    },
    'nlvr2': {
        'annotation': 'dataset/NLVR2',
        'ds_prompt': "Left image: <|image|> Right image: <|image|>\n{question}\nyes or no? Answer the question using a single word.",
        'only_submit': False
    },
    'mi_bench': {
        'annotation': 'path/to/MIBench',
        'ds_prompt': "{question}\nOptions: {options}\nAnswer with the option’s letter from the given choices directly.",
        'only_submit': False
    },
    'ai2d': {
        'annotation': 'dataset/ai2d/',
        'ds_prompt': "<|image|>\nQuestion: {question}\nOptions:\n{options}\nAnswer with the option’s number from the given choices directly.",
        'ds_prompt_2': "<|image|>\nQuestion: {question}\nOptions:\n{options}\nAnswer with the option’s letter from the given choices directly.",
        'only_submit': False
    },
    'mmvet': {
        'annotation': 'dataset/mmvet/mm-vet.json',
        'image_path': 'images/mmvet/images/',
        'ds_prompt': "<|image|>\n{question}",
        'only_submit': True
    },
}

def extract_choice(model_output: str) -> str:
    # Define the possible patterns for choices
    # model_output = model_output.replace("There are several options:\nA.", '')
    patterns = [
        r"\b{}[.,:)]",  # Choice followed by a delimiter (.,))
        r"\b{}[.,:)]\b",  # Choice surrounded by delimiters (.,))
    ]

    # Iterate through choices A, B, C, D, and E
    for choice in "EDCBA":
        for pattern in patterns:
            # Check if the choice pattern is found in the model_output
            match = re.search(pattern.format(choice), model_output)
            if match:
                return choice  # Return the matched choice

    return None  # If no choice is found, return None


def extract_answer(model_output, options):
    res = extract_choice(model_output)
    if res is None:
        res = extract_choice(model_output + '.')
    
    if res is None:
        letters = "ABCDEFG"
        for i, option in enumerate(options):
            if model_output.lower() in option.lower() or option.lower() in model_output.lower():
                res = letters[i]
                break
    if res is None:
        res = "X"
    return res

def calculate_metric(results):
    l2_category = defaultdict(int)
    l2_category_correct = defaultdict(int)
    for result in results:
        result['extracted_prediction'] = extract_answer(result['prediction'], result['options'])
    res_df = pd.DataFrame(results)
    groups = res_df.groupby(['image', 'question'])
    result_groups = [group.to_dict(orient='records') for _, group in groups]
    for sample in result_groups:
        category = sample[0]['l2-category']
        l2_category[category] += 1
        cnt = 0
        for s in sample:
            if s['extracted_prediction'] == s['answer']:
                cnt += 1
        if cnt == len(sample):
            l2_category_correct[category] += 1    
                    
    l2_category_acc = {k: l2_category_correct[k] / l2_category[k] * 100 for k in l2_category}
    total_acc = sum(l2_category_correct.values()) / sum(l2_category.values()) * 100
    print(l2_category_acc)
    print("Average Score:", sum(l2_category_correct.values()) / sum(l2_category.values()) * 100)
    return {"Accuracy": total_acc, "L2-Accuracy": l2_category_acc}

def mapping_to_annotation(results, raw_annotation):
    outputs = []
    for result in results:
        index, prediction = result['index'], result['prediction']
        row_df = raw_annotation[raw_annotation['index'] == index].squeeze().to_dict()
        output = {
            "index": index,
            "image": row_df['image'],
            "question": row_df['question'],
            "answer": row_df.get('answer', None),
            "options": [y for y in [row_df.get(x, None) for x in 'ABCD'] if isinstance(y, str)],
            "prediction": prediction,
            "l2-category": row_df['l2-category'] if 'l2-category' in row_df else None 
        }
        outputs.append(output)
    return outputs

def generate_submission_file(results, raw_annotation):
    outputs = []
    for result in results:
        index, prediction = result['index'], result['prediction']
        row_df = raw_annotation[raw_annotation['index'] == index].squeeze().to_dict()
        output = {
            "index": index,
            "question": row_df['question'],
            "prediction": prediction,
            "A": row_df.get('A', None),
            "B": row_df.get('B', None),
            "C": row_df.get('C', None),
            "D": row_df.get('D', None),
        }
        outputs.append(output)
    return outputs


@TASK.register_module()
class MultiOptionTask(EvalTaskBase):
    def __init__(self, ds_name, **kwargs) -> None:
        task_config = ds_collections[ds_name]
        super().__init__(task_config, **kwargs)

        # pre setting
        self.ds_name = ds_name
        self.only_submit = task_config['only_submit']

        self.ds_anno = task_config['annotation']
        self.datas = read_json(self.ds_anno)
        self.prompt = '<|image|>Context: {}\nQuestion: {}\nOptions: {}\nAnswer with the option’s letter from the given choices directly.'
        self.image_path = task_config['image_path']

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        data = self.datas[index]
        image = os.path.join(self.image_path, data['image'])
        # image = 'images/' + data['image']
        # image = data['image']
        hint = data['hint'] if data['hint'] else 'N/A'
        images = [image]
        question = data['question']

        choices = data['choices']

        choice_list = []
        for i, c in enumerate(choices):
            choice_list.append('{}. {}'.format(multiple_choices[i], c))
        choice_txt = '\n'.join(choice_list)

        prompt = self.prompt.format(hint, question, choice_txt)

        messages = [
            {"role": "system", "content": self.system_prompt if self.enable_system_prompt else ""},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "None"}
        ]
        if 'answer' in data:
            gt_answer = multiple_choices[data['answer']]
        else:
            gt_answer = 'N/A'

        return StandardData(messages=messages, images=images, gt_answer=gt_answer, index=index, extra={"question_id":data['index']})


@TASK.register_module()
class MMBenchTask(MultiOptionTask):
        
    def postprocess(self, line: StandardData):
        output = {
            "index": line.extra['question_id'],
            "prediction": line.raw_model_answer,
            "ground_truth": line.gt_answer
        }
        return output
        
    def build_submission(self, merged_outputs, saver):
        submission_res = generate_submission_file(merged_outputs, pd.read_csv(self.task_config['raw_file'], sep='\t'))
        res_df = pd.DataFrame(submission_res)

        metrics_file = f'{saver.get_save_dir()}/{self.ds_name}_{saver.time_prefix}_submission.xlsx'
        res_df.to_excel(metrics_file, index=False)
        return metrics_file

    def evaluate(self, merged_outputs, results_file=None, submission_file=None):
        mapped_result = mapping_to_annotation(merged_outputs, pd.read_csv(self.task_config['raw_file'], sep='\t'))
        if not self.only_submit:
            metrics = calculate_metric(mapped_result)
        else:
            metrics = {}
        if submission_file is not None:
            metrics['submission_file'] = submission_file
        return metrics, merged_outputs


@TASK.register_module()
class MIBench(EvalTaskBase):
    def __init__(self, ds_name, **kwargs) -> None:
        task_config = ds_collections[ds_name]
        EvalTaskBase.__init__(self, task_config, **kwargs)

        # pre setting
        self.ds_name = ds_name
        self.only_submit = task_config['only_submit']

        self.ds_anno = task_config['annotation']
        self.prompt = task_config['ds_prompt']
        self.datas = []
        for task in ['multi_image_instruction', 'multimodal_knowledge_seeking']:
            lines = read_json(Path(self.ds_anno, f'{task}.json'))
            for line in lines:
                line['category'] = task
                len_ops = len(line['options'])
                for i in range(len_ops):
                    rline = copy.deepcopy(line)
                    self.datas.append(rline)
                    line['options'] = line['options'][1:] + line['options'][:1]
                    
    def __len__(self,):
        return len(self.datas)

    def __getitem__(self, index):
        line = self.datas[index]

        question = line['question']
        options = line['options']
        gt_answer = f"{chr(ord('A')+options.index(line['answer']))}"
        options = '\n'.join([f"{chr(ord('A')+i)}. {opt}" for i,opt in enumerate(options)])
        messages = [
            {"role": "user", "content": self.prompt.format(question=question.replace('<image>','<|image|>'), options=options)},
            {"role": "assistant", "content":""}
        ]

        images = [str(Path(self.ds_anno, _)) for _ in line['image']]
        return StandardData(messages=messages, images=images, gt_answer=gt_answer, index=index, extra=line)

    def postprocess(self, line: StandardData):
        output = {
            "index": line.extra['id'],
            "prediction": line.raw_model_answer,
            "ground_truth": line.gt_answer,
            "category": line.extra['task'],
            "category_top": line.extra['category']
        }
        return output
    
    def evaluate(self, merged_outputs, results_file=None, submission_file=None):
        qid2answers = defaultdict(list)
        for line in merged_outputs:
            qid2answers[line['index']].append(line)

        accuracy_category = {
            'multi_image_instruction': defaultdict(list),
            'multimodal_knowledge_seeking': defaultdict(list),
        }
        overall_acc = []
        for qid, pred_list in qid2answers.items():
            category = pred_list[0]['category']
            category_top = pred_list[0]['category_top']
            accuracy_category[category_top][category].append(all([_['prediction'].lower()[0] == _['ground_truth'].lower() for _ in pred_list]))
            overall_acc.append(accuracy_category[category_top][category][-1])
        metrics = {
            "Overall-Accuracy": sum(overall_acc)/len(overall_acc),
            "Category":{
                k: {c: sum(acc)/len(acc) for c,acc in v.items()}
                for k,v in accuracy_category.items()
            }
        }
        return metrics, merged_outputs

@TASK.register_module()
class AI2DBench(EvalTaskBase):
    def __init__(self, ds_name, use_number=True, **kwargs) -> None:
        task_config = ds_collections[ds_name]
        EvalTaskBase.__init__(self, task_config, **kwargs)

        # pre setting
        self.ds_name = ds_name
        self.only_submit = task_config['only_submit']
        self.use_number = use_number
        self.datas = load_dataset(task_config['annotation'])['test']
        if self.use_number:
            self.prompt = task_config['ds_prompt']
        else:
            self.prompt = task_config['ds_prompt_2']

    def __len__(self,):
        return len(self.datas)
        
    def __getitem__(self, index):
        line = self.datas[index]
        images = [line['image']]
        question = line['question']
        if self.use_number:
            options = '\n'.join([f"{oi}. {o}" for oi,o in enumerate(line['options'])])
        else:
            options = '\n'.join([f"{multiple_choices[oi]}. {o}" for oi,o in enumerate(line['options'])])
        messages = [
            {"role": "user", "content": self.prompt.format(question=question, options=options)},
            {"role": "assistant", "content":""}
        ]
        if self.use_number:
            gt_answer = line['answer']
        else:
            gt_answer = multiple_choices[int(line['answer'])]

        return StandardData(messages=messages, images=images, videos=[], gt_answer=gt_answer, index=index, extra=line)
    
    def postprocess(self, line: StandardData):
        output = {
            "pred": line.raw_model_answer,
            "ground_truth": line.gt_answer,
        }
        return output
    
    def evaluate(self, merged_outputs, results_file=None, submission_file=None):
        metrics = {
            'Accuracy': np.mean([_['pred'][0] == _['ground_truth'][0] for _ in merged_outputs])
        }
        return metrics, merged_outputs

@TASK.register_module()
class NLVR2Bench(EvalTaskBase):
    def __init__(self, ds_name, use_number=True, **kwargs) -> None:
        task_config = ds_collections[ds_name]
        EvalTaskBase.__init__(self, task_config, **kwargs)

        # pre setting
        self.ds_name = ds_name
        self.only_submit = task_config['only_submit']
        self.use_number = use_number
        self.datas = load_dataset(task_config['annotation'])['unbalanced_test_public']
        self.prompt = task_config['ds_prompt']
    def __len__(self,):
        return len(self.datas)
        
    def __getitem__(self, index):
        line = self.datas[index]
        images = [line['left_image'], line['right_image']]
        question = line['question']
       
        messages = [
            {"role": "user", "content": self.prompt.format(question=question)},
            {"role": "assistant", "content":""}
        ]
        gt_answer = 'no' if line['answer'] == 'False' else 'yes'

        return StandardData(messages=messages, images=images, videos=[], gt_answer=gt_answer, index=index, extra=line)
    
    def postprocess(self, line: StandardData):
        output = {
            "pred": line.raw_model_answer,
            "ground_truth": line.gt_answer,
        }
        return output
    
    def evaluate(self, merged_outputs, results_file=None, submission_file=None):
        metrics = {
            'Accuracy': np.mean([_['pred'].split('.')[0].split(',')[0].lower().strip() == _['ground_truth'].lower() for _ in merged_outputs])
        }
        return metrics, merged_outputs

@TASK.register_module()
class MMVetBench(EvalTaskBase):
    def __init__(self, ds_name, **kwargs) -> None:
        task_config = ds_collections[ds_name]
        EvalTaskBase.__init__(self, task_config, **kwargs)

        # pre setting
        self.ds_name = ds_name
        self.only_submit = task_config['only_submit']
        self.datas = json.load(open(task_config['annotation']))
        self.prompt = task_config['ds_prompt']
        self.image_path = task_config['image_path']
    def __len__(self,):
        return len(self.datas)
        
    def __getitem__(self, index):
        # line = self.datas[index]
        question_id = f"v1_{index}"
        d = self.datas[question_id]
        image_name, question = d['imagename'], d['question'].strip()
        
        images = [os.path.join(self.image_path, image_name)]
       
        messages = [
            {"role": "user", "content": self.prompt.format(question=question)},
            {"role": "assistant", "content":""}
        ]
        gt_answer = 'None'
        d['question_id'] = question_id
        return StandardData(messages=messages, images=images, videos=[], gt_answer=gt_answer, index=index, extra=d)
    
    def postprocess(self, line: StandardData):
        output = {
            'answer': line.raw_model_answer,
            'question_id': line.extra['question_id'],
        }
        return output
    
    def build_submission(self, merged_outputs, saver):
        submission = {}
        for output in merged_outputs:
            submission[output['question_id']] = output['answer']
        metrics_file = f'{saver.get_save_dir()}/{self.ds_name}_{saver.time_prefix}_submit.json'
        json.dump(submission, open(metrics_file, 'w'), ensure_ascii=False) # save to results
        return metrics_file
    
    def evaluate(self, merged_outputs, results_file=None, submission_file=None):
        metrics = {}
        if self.only_submit:
            metrics['submission_file'] = submission_file
        return metrics, merged_outputs

class MMBenchResulterSaver(ResulterSaverBase):
   

    def save_metrics(self, metrics):
        if self.time_prefix is None:
            self.time_prefix = time.strftime("%y%m%d%H%M%S", time.localtime())
        
        metrics_file = f'{self.get_save_dir()}_{self.time_prefix}_submission.xlsx'
        

        metrics_file = f'{self.get_save_dir()}_{self.time_prefix}_metrics.json'
        write_json(metrics, metrics_file)
        return metrics_file

