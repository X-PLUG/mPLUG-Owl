import ast
import copy
import os

from tasks.task_define.mmbench import MultiOptionTask
from .base import EvalTaskBase, TASK, ResulterSaverBase, StandardData, load_jsonl
from x.io import read_json
from icecream import ic
from datasets import load_dataset
from pathlib import Path

from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import torch

import json
map_name = {'CW': 'Why', 'CH': 'How', 'TN': 'Bef&Aft', 'TC': 'When', 'DC': 'Cnt', 'DL': 'Loc', 'DO': 'Other', 'C': 'Acc_C', 'T': 'Acc_T', 'D': 'Acc_D'}


ds_collections = {
    "next_qa_mc": {
        'path': 'dataset/nextqa/',
        'split': 'MC',
        'ds_prompt': "<|video|>\nQuestion: {question}\nOptions: {options}\nAnswer with the option’s letter from the given choices directly.",
        'only_submit': False,
    },
    'videomme': {
        'path': "dataset/videomme",
        'ds_prompt': "<|video|>\nQuestion: {question}\nOptions: {options}\nAnswer with the option’s letter from the given choices directly.",
        'only_submit': False,
    },
    'long_video_bench_dev': {
        'path': "dataset/LongVideoBench",
        'ds_prompt': "Question: {question}\nOptions: {options}\nAnswer with the option’s letter from the given choices directly.",
        'split': 'lvb_val',
        'only_submit': False,
    }
}


@TASK.register_module()
class NextQATask(MultiOptionTask):
    def __init__(self, ds_name, num_frames=8, **kwargs) -> None:
        task_config = ds_collections[ds_name]
        EvalTaskBase.__init__(self, task_config, **kwargs)

        # pre setting
        self.ds_name = ds_name
        self.only_submit = task_config['only_submit']

        self.datas = load_dataset(task_config['path'],task_config['split'])['test']
        self.media_root = Path(task_config['path'], 'NExTVideo')
        self.prompt = task_config['ds_prompt']
        self.num_frames = num_frames


    def __getitem__(self, index):
        line = self.datas[index]
        video_path = str(Path(self.media_root,f"{line['video']}.mp4"))
        videos = [{'video':video_path, 'num_frames': self.num_frames}]
        question = line['question']
        options = '\n'.join([f"{chr(ord('A')+i)}. {line[f'a{i}']}" for i in range(10) if f"a{i}" in line])
        messages = [
            {"role": "user", "content": self.prompt.format(question=question, options=options)},
            {"role": "assistant", "content":""}
        ]
        gt_answer = f"{chr(ord('A')+line['answer'])}"

        return StandardData(messages=messages, images=[], videos=videos, gt_answer=gt_answer, index=index, extra=line)

    def postprocess(self, line: StandardData):
        if len(line.raw_model_answer) == 0:
            line.raw_model_answer = 'X'
        output = {
            "raw": line.extra,
            "pred": line.raw_model_answer,
            "ground_truth": line.gt_answer,
        }
        return output
    
    def evaluate(self, merged_outputs, results_file=None, submission_file=None):
        preds = {}
        for line in merged_outputs:
            qns_id = str(line['raw']['video']) + '_' + str(line['raw']['qid'])
            preds[qns_id] = {
                'answer': f"{chr(ord('A')+line['raw']['answer'])}".lower(),
                'prediction': line['pred'][0].strip().lower()[0],
            }
        metrics = accuracy_metric(sample_list=self.datas, preds=preds)
        return metrics, merged_outputs

def accuracy_metric(sample_list, preds):

    group = {'CW':[], 'CH':[], 'TN':[], 'TC':[], 'DC':[], 'DL':[], 'DO':[]}
    for row in sample_list:
        qns_id = str(row['video']) + '_' + str(row['qid'])
        qtype = str(row['type'])
        #(combine temporal qns of previous and next as 'TN')
        if qtype == 'TP': qtype = 'TN'
        group[qtype].append(qns_id)

    group_acc = {'CW': 0, 'CH': 0, 'TN': 0, 'TC': 0, 'DC': 0, 'DL': 0, 'DO': 0}
    group_cnt = {'CW': 0, 'CH': 0, 'TN': 0, 'TC': 0, 'DC': 0, 'DL': 0, 'DO': 0}
    overall_acc = {'C':0, 'T':0, 'D':0}
    overall_cnt = {'C':0, 'T':0, 'D':0}
    all_acc = 0
    all_cnt = 0
    for qtype, qns_ids in group.items():
        cnt = 0
        acc = 0
        for qid in qns_ids:

            cnt += 1
            answer = preds[qid]['answer']
            pred = preds[qid]['prediction']

            if answer == pred: 
                acc += 1

        group_cnt[qtype] = cnt
        group_acc[qtype] += acc
        overall_acc[qtype[0]] += acc
        overall_cnt[qtype[0]] += cnt
        all_acc += acc
        all_cnt += cnt


    for qtype, value in overall_acc.items():
        group_acc[qtype] = value
        group_cnt[qtype] = overall_cnt[qtype]

    for qtype in group_acc:
        print(map_name[qtype], end='\t')
    print('')
    metrics = {}
    for qtype, acc in group_acc.items():
        print('{:.2f}'.format(acc*100.0/group_cnt[qtype]), end ='\t')
        metrics[qtype] = acc*100.0/group_cnt[qtype]
    print('')
    print('Acc: {:.2f}'.format(all_acc*100.0/all_cnt))
    metrics = {
        'Accuracy': all_acc*100.0/all_cnt,
        'category': metrics
    }
    return metrics



def evaluate_exact_match_accuracy(entries, post_process=False):
    scores = []
    for elem in entries:
        if isinstance(elem['annotation'], str):
            elem['annotation'] = [elem['annotation']]
        if post_process:
            manualMap = {
                "none": "0",
                "zero": "0",
                "one": "1",
                "two": "2",
                "three": "3",
                "four": "4",
                "five": "5",
                "six": "6",
                "seven": "7",
                "eight": "8",
                "nine": "9",
                "ten": "10",
            }
            elem['answer'] = elem['answer'].split('.')[0].split(',')[0].strip()
            elem['annotation'] = [x if x not in manualMap else manualMap[x] for x in elem['annotation']]
        score = max([
            (1.0 if
             (elem['answer'].strip().lower() == ann.strip().lower()) else 0.0)
            for ann in elem['annotation']
        ])
        scores.append(score)
    return sum(scores) / len(scores)

def evaluate_in_match_accuracy(entries, post_process=False):
    scores = []
    for elem in entries:
        if isinstance(elem['annotation'], str):
            elem['annotation'] = [elem['annotation']]
        if post_process:
            manualMap = {
                "none": "0",
                "zero": "0",
                "one": "1",
                "two": "2",
                "three": "3",
                "four": "4",
                "five": "5",
                "six": "6",
                "seven": "7",
                "eight": "8",
                "nine": "9",
                "ten": "10",
            }
            elem['answer'] = elem['answer'].split('.')[0].split(',')[0].strip()
            elem['annotation'] = [x if x not in manualMap else manualMap[x] for x in elem['annotation']]
        score = max([
            (1.0 if
             (elem['answer'].strip().lower() in ann.strip().lower() or ann.strip().lower() in elem['answer'].strip().lower()) else 0.0)
            for ann in elem['annotation']
        ])
        scores.append(score)
    return sum(scores) / len(scores)

@TASK.register_module()
class VideoQA(EvalTaskBase):
    def __init__(self, ds_name, num_frames=16, **kwargs) -> None:
        task_config = ds_collections[ds_name]
        EvalTaskBase.__init__(self, task_config, **kwargs)
        self.ds_name = ds_name
        self.path = task_config['path']
        self.num_frames = num_frames
        self.prompt = task_config['ds_prompt']
     
        self.datas = read_json(self.path)
    
    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = self.datas[idx]
        image_id, image_path, question_id = data['video'], os.path.join(data['image_root'], data['video']), data['data_id']
        question = data['query']
        annotation = data["answer"]

        videos = [{'video': image_path, 'num_frame': self.num_frames}]
        messages = [
            {"role": "user", "content": self.prompt.format(question=question)},
            {"role": "assistant", "content":""}
        ]
        gt_answer = annotation
        return StandardData(messages=messages, images=[], videos=videos, gt_answer=gt_answer, index=idx, extra=data)


    def postprocess(self, line: StandardData):
        output = {
                "question_id": line.extra['data_id'],
                "answer": line.raw_model_answer,
                "annotation": line.gt_answer
            }
        return output

    def evaluate(self, merged_outputs, results_file=None, submission_file=None):
        metrics = {
            "Exact Accuracy": evaluate_exact_match_accuracy(merged_outputs) * 100,
            "Exact Accuracy (Post)": evaluate_exact_match_accuracy(merged_outputs, post_process=True) * 100,
            "In Accuracy": evaluate_in_match_accuracy(merged_outputs) * 100,
            "In Accuracy (Post)": evaluate_in_match_accuracy(merged_outputs, post_process=True) * 100
        }
        return metrics, merged_outputs

@TASK.register_module()
class VideoMMEBench(EvalTaskBase):
    def __init__(self, ds_name, num_frames=16, **kwargs) -> None:
        task_config = ds_collections[ds_name]
        EvalTaskBase.__init__(self, task_config, **kwargs)

        # pre setting
        self.ds_name = ds_name
        self.num_frames = num_frames
        self.only_submit = task_config['only_submit']
        self.datas = load_dataset(task_config['path'])['test']
        self.media_root = 'dataset/videomme/data'
        self.prompt = task_config['ds_prompt']

        self.CATEGORIES = [
            "Knowledge",
            "Film & Television",
            "Sports Competition",
            "Artistic Performance",
            "Life Record",
            "Multilingual"
        ]

        self.SUB_CATEGORIES = [
            "Humanity & History",
            "Literature & Art",
            "Biology & Medicine",
            "Finance & Commerce",
            "Astronomy",
            "Geography",
            "Law",
            "Life Tip",
            "Technology",
            "Animation",
            "Movie & TV Show",
            "Documentary",
            "News Report",
            "Esports",
            "Basketball",
            "Football",
            "Athletics",
            "Other Sports",
            "Stage Play",
            "Magic Show",
            "Variety Show",
            "Acrobatics",
            "Handicraft",
            "Food",
            "Fashion",
            "Daily Life",
            "Travel",
            "Pet & Animal",
            "Exercise",
            "Multilingual"
        ]

        self.TASK_CATEGORIES = [
            "Temporal Perception",
            "Spatial Perception",
            "Attribute Perception",
            "Action Recognition",
            "Object Recognition",
            "OCR Problems",
            "Counting Problem",
            "Temporal Reasoning",
            "Spatial Reasoning",
            "Action Reasoning",
            "Object Reasoning",
            "Information Synopsis",
        ]
    def __len__(self):
        return len(self.datas)
    
    def extract_characters_regex(self, s):
        import re
        s = s.strip()
        answer_prefixes = [
            "The best answer is",
            "The correct answer is",
            "The answer is",
            "The answer",
            "The best option is"
            "The correct option is",
            "Best answer:"
            "Best option:",
            "Answer:",
            "Option:",
            "The correct answer",
            "The correct option",
        ]
        for answer_prefix in answer_prefixes:
            s = s.replace(answer_prefix, "")

        if len(s.split()) > 10 and not re.search("[ABCD]", s):
            return "A"
        matches = re.search(r'[ABCD]', s)
        if matches is None:
            return "A"
        return matches[0]

    def __getitem__(self, index):
        line = self.datas[index]
        video_path = str(Path(self.media_root,f"{line['videoID']}.mp4"))
        videos = videos = [{'video':video_path, 'num_frames': self.num_frames}]
        question = line['question']
        options = '\n'.join(line['options'])
        messages = [
            {"role": "user", "content": self.prompt.format(question=question, options=options)},
            {"role": "assistant", "content":""}
        ]
        gt_answer = line['answer']

        return StandardData(messages=messages, images=[], videos=videos, gt_answer=gt_answer, index=index, extra=line)

    def postprocess(self, line: StandardData):
        # output = {
        #     "raw": line.extra,
        #     "pred": line.raw_model_answer,
        #     "ground_truth": line.gt_answer,
        # }
        output = copy.deepcopy(line.extra)
        output['pred'] = line.raw_model_answer
        return output
    
    def evaluate(self, merged_outputs, results_file=None, submission_file=None):

        video_types = ["short","medium","long"]
        q_type_dict = {d:{q_type: {"correct": 0, "answered": 0} for q_type in self.TASK_CATEGORIES}  for d in video_types}
        v_type_dict = {d:{v_type: {"correct": 0, "answered": 0} for v_type in self.CATEGORIES}  for d in video_types}
        v_sub_type_dict = {d:{v_sub_type: {"correct": 0, "answered": 0} for v_sub_type in self.SUB_CATEGORIES}  for d in video_types}


        for item in merged_outputs:
            video_type = item['duration']
            video_category = item["domain"]
            video_sub_category = item["sub_category"]
            q_type = item["task_type"]
            gt_answer = item['answer']
            response = item['pred']

            # Extract the answer from the response
            extration = self.extract_characters_regex(response)

            q_type_dict[video_type][q_type]["answered"] += 1
            q_type_dict[video_type][q_type]["correct"] += extration == gt_answer

            v_type_dict[video_type][video_category]["answered"] += 1
            v_type_dict[video_type][video_category]["correct"] += extration == gt_answer

            v_sub_type_dict[video_type][video_sub_category]["answered"] += 1
            v_sub_type_dict[video_type][video_sub_category]["correct"] += extration == gt_answer
                # Get the video category, sub category and question category
                
        metrics = {}

        # Print the results for each video type
        metrics['Duration'] = {}
        for video_type in video_types:

            print("=====================================")
            print(f"Evaluation on video Type: {video_type}")
            print("=====================================")
            print("-------------------------------------")
            print("Video Categories")
            print("-------------------------------------")
            for v_type in v_type_dict[video_type]:
                print(f"{v_type}: {100 * v_type_dict[video_type][v_type]['correct'] / v_type_dict[video_type][v_type]['answered'] if v_type_dict[video_type][v_type]['answered'] > 0 else 0 : .1f}%")
            print("-------------------------------------")
            print("Video Sub Categories")
            print("-------------------------------------")
            for v_sub_type in v_sub_type_dict[video_type]:
                print(f"{v_sub_type}: {100 * v_sub_type_dict[video_type][v_sub_type]['correct'] / v_sub_type_dict[video_type][v_sub_type]['answered'] if v_sub_type_dict[video_type][v_sub_type]['answered'] > 0 else 0 : .1f}%")
            print("-------------------------------------")
            print("Task Categories")
            print("-------------------------------------")
            for q_type in q_type_dict[video_type]:
                print(f"{q_type}: {100 * q_type_dict[video_type][q_type]['correct'] / q_type_dict[video_type][q_type]['answered'] if q_type_dict[video_type][q_type]['answered'] > 0 else 0 : .1f}%")
            
            print("-------------------------------------")
            print("Overall Performance")
            print("-------------------------------------")
            total_correct = sum([q_type_dict[video_type][q_type]["correct"] for q_type in self.TASK_CATEGORIES])
            total_answered = sum([q_type_dict[video_type][q_type]["answered"] for q_type in self.TASK_CATEGORIES])
            print(f"Overall: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")
            metrics[video_type] = f"{100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%"
            print("\n")

        # Print the results for the entire dataset
        print("=====================================")
        print("Evaluation on the entire dataset")
        print("=====================================")

        print("-------------------------------------")
        print("Video Domains")
        print("-------------------------------------")
        for v_type in self.CATEGORIES:
            total_correct = sum([v_type_dict[video_type][v_type]["correct"] for video_type in video_types])
            total_answered = sum([v_type_dict[video_type][v_type]["answered"] for video_type in video_types])
            print(f"{v_type}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")
        

        print("-------------------------------------")
        print("Video Sub Categories")
        print("-------------------------------------")

        for v_sub_type in self.SUB_CATEGORIES:
            total_correct = sum([v_sub_type_dict[video_type][v_sub_type]["correct"] for video_type in video_types])
            total_answered = sum([v_sub_type_dict[video_type][v_sub_type]["answered"] for video_type in video_types])
            print(f"{v_sub_type}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")


        print("-------------------------------------")
        print("Task Categories")
        print("-------------------------------------")
        for q_type in self.TASK_CATEGORIES:

            total_correct = sum([q_type_dict[video_type][q_type]["correct"] for video_type in video_types])
            total_answered = sum([q_type_dict[video_type][q_type]["answered"] for video_type in video_types])
            print(f"{q_type}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

        print("-------------------------------------")
        print("Overall Performance")
        print("-------------------------------------")
        total_correct = sum([sum([q_type_dict[video_type][q_type]["correct"] for q_type in self.TASK_CATEGORIES]) for video_type in video_types])
        total_answered = sum([sum([q_type_dict[video_type][q_type]["answered"] for q_type in self.TASK_CATEGORIES]) for video_type in video_types])
        print(f"Overall: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")
        metrics['Accuracy Overall'] = f"{100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%"
       
        return metrics, merged_outputs

def check_answer(predict, gt_answer):
    flag = False

    predict = predict.lower()
    gt_answer = gt_answer.lower()
    
    pred_list = predict.lower().split(' ')
    pred_option, pred_content = pred_list[0], ' '.join(pred_list[1:])
    
    gt_list = gt_answer.lower().split(' ')
    gt_option, gt_content = gt_list[0], ' '.join(gt_list[1:])
    if gt_content[-1] == '.':
        gt_content = gt_content[:-1]
    
    if pred_option.replace('.', '').strip() in gt_option:
        flag = True
    elif gt_content.strip() in pred_content:
        flag = True
        
    return flag

@TASK.register_module()
class LongVideoBenchTask(EvalTaskBase):
    def __init__(self, ds_name, num_frames=16, **kwargs) -> None:
        task_config = ds_collections[ds_name]
        EvalTaskBase.__init__(self, task_config, **kwargs)
        self.ds_name = ds_name
        self.path = task_config['path']
        self.num_frames = num_frames
        self.prompt = task_config['ds_prompt']
     
        self.datas = LongVideoBenchDataset(task_config['path'], f"{task_config['split']}.json", max_num_frames=num_frames) # default 64
    
    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        line = self.datas[index]
        if line['video']:
            if line['bound']:
                videos = [{'video': line['video'], 'bound': line['bound'], 'num_frames': self.num_frames}]
            else:
                videos = [{'video':line['video'], 'num_frames': self.num_frames}]
            question = '<|video|>' + '\n' + line['question']
            frames = []
        else:
            frames = line['image']
            videos = []
            question = '<|image|>'*len(line['image']) + '\n' + line['question']

        
        # options = [f"{chr(ord('A')+i)}. {line[f'a{i}']}" for i in range(10) if f"a{i}" in line]
        gt_answer = line['answer']
        options = '\n'.join(line['options'])
        messages = [
            {"role": "user", "content": self.prompt.format(question=question, options=options)},
            {"role": "assistant", "content":""}
        ]

        return StandardData(messages=messages, images=frames, videos=videos, gt_answer=gt_answer, index=index, extra=line)

    def postprocess(self, line: StandardData):
        output = {
            "raw": line.extra,
            "pred": line.raw_model_answer,
            "ground_truth": line.gt_answer,
        }
        return output
    
    def evaluate(self, merged_outputs, results_file=None, submission_file=None):
        all_acc = []
        for line in merged_outputs:
            all_acc.append(check_answer(line['pred'], line['ground_truth']))
        metrics = {}
        
        metrics['Accuracy_overall'] = np.mean(all_acc)
        print(metrics)
        return metrics, merged_outputs


class LongVideoBenchDataset(Dataset):
    def __init__(self,
                 data_path,
                 annotation_file,
                 max_num_frames=256,
                 insert_text=True,
                 insert_frame=True,
                ):
        super().__init__()
        self.data_path = data_path
        self.insert_text = insert_text

        with open(os.path.join(data_path, annotation_file)) as f:
            self.data = json.load(f)
        self.max_num_frames = max_num_frames
        
        
        
    def __getitem__(self, index):
        di = self.data[index]
        
        return {
            'video': os.path.join(self.data_path, "videos", di["video_path"]),
            'bound': [0 ,di["duration"]],
            'question': di["question"].replace('<|video|>',''),
            'options': [chr(ord("A")+i)+'. '+ candidate for i, candidate in enumerate(di["candidates"])],
            'answer': chr(ord("A")+di.get("correct_choice", -1))+'. '+di["candidates"][di.get("correct_choice", 0)],
            "id": di["id"],
        }
    
    def __len__(self):
        return len(self.data)
    
    def get_id(self, index):
        return self.data[index]["id"]
        