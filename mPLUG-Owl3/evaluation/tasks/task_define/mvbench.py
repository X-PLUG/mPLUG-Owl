import ast
import copy
import os

from tasks.task_define.mmbench import MultiOptionTask
from .base import EvalTaskBase, TASK, ResulterSaverBase, StandardData, load_jsonl
from x.io import read_json
from icecream import ic
from datasets import load_dataset
from pathlib import Path
from collections import defaultdict
map_name = {'CW': 'Why', 'CH': 'How', 'TN': 'Bef&Aft', 'TC': 'When', 'DC': 'Cnt', 'DL': 'Loc', 'DO': 'Other', 'C': 'Acc_C', 'T': 'Acc_T', 'D': 'Acc_D'}


ds_collections = {
    "mvbench": {
        'path': 'dataset/mvbench',
        'ds_prompt': "{question}\nAnswer with the option’s letter from the given choices directly.",
        'only_submit': False,
    },
}


from datasets import load_dataset
splits = ['action_sequence', 'moving_count', 'action_prediction', 'episodic_reasoning', 'action_antonym', 'action_count', 'scene_transition', 'object_shuffle', 'object_existence', 'fine_grained_pose', 'unexpected_action', 'moving_direction', 'state_change', 'object_interaction', 'character_order', 'action_localization', 'counterfactual_inference', 'fine_grained_action', 'moving_attribute', 'egocentric_navigation']
data_list = {
    "Action Sequence": ("action_sequence.json", "your_data_path/star/Charades_v1_480/", "video", True), # has start & end
    "Action Prediction": ("action_prediction.json", "your_data_path/star/Charades_v1_480/", "video", True), # has start & end
    "Action Antonym": ("action_antonym.json", "your_data_path/ssv2_video/", "video", False),
    "Fine-grained Action": ("fine_grained_action.json", "your_data_path/Moments_in_Time_Raw/videos/", "video", False),
    "Unexpected Action": ("unexpected_action.json", "your_data_path/FunQA_test/test/", "video", False),
    "Object Existence": ("object_existence.json", "your_data_path/clevrer/video_validation/", "video", False),
    "Object Interaction": ("object_interaction.json", "your_data_path/star/Charades_v1_480/", "video", True), # has start & end
    "Object Shuffle": ("object_shuffle.json", "your_data_path/perception/videos/", "video", False),
    "Moving Direction": ("moving_direction.json", "your_data_path/clevrer/video_validation/", "video", False),
    "Action Localization": ("action_localization.json", "your_data_path/sta/sta_video/", "video", True),  # has start & end
    "Scene Transition": ("scene_transition.json", "your_data_path/scene_qa/video/", "video", False),
    "Action Count": ("action_count.json", "your_data_path/perception/videos/", "video", False),
    "Moving Count": ("moving_count.json", "your_data_path/clevrer/video_validation/", "video", False),
    "Moving Attribute": ("moving_attribute.json", "your_data_path/clevrer/video_validation/", "video", False),
    "State Change": ("state_change.json", "your_data_path/perception/videos/", "video", False),
    "Fine-grained Pose": ("fine_grained_pose.json", "your_data_path/nturgbd/", "video", False),
    "Character Order": ("character_order.json", "your_data_path/perception/videos/", "video", False),
    "Egocentric Navigation": ("egocentric_navigation.json", "your_data_path/vlnqa/", "video", False),
    "Episodic Reasoning": ("episodic_reasoning.json", "your_data_path/tvqa/frames_fps3_hq/", "frame", True),  # has start & end, read frame
    "Counterfactual Inference": ("counterfactual_inference.json", "your_data_path/clevrer/video_validation/", "video", False),
}
data_list = {k.lower().replace(' ','_').replace('-','_'):v for k,v in data_list.items()}


@TASK.register_module()
class MvBenchTask(MultiOptionTask):
    def __init__(self, ds_name, num_frames=16, **kwargs) -> None:
        task_config = ds_collections[ds_name]
        EvalTaskBase.__init__(self, task_config, **kwargs)

        # pre setting
        self.ds_name = ds_name
        self.num_frames = num_frames
        self.only_submit = task_config['only_submit']
        from torch.utils.data import ConcatDataset
        self.datas = ConcatDataset([MVBench_dataset(task_config['path'],split, num_frames=num_frames) for split in data_list])
        self.prompt = task_config['ds_prompt']

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

        
        gt_answer = line['answer']
    
        messages = [
            {"role": "user", "content": self.prompt.format(question=question)},
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
        acc = defaultdict(list)
        all_acc = []
        for line in merged_outputs:
            acc[line['raw']['category']].append(check_answer(line['pred'], line['ground_truth']))
            all_acc.append(check_answer(line['pred'], line['ground_truth']))
        metrics = {k:np.mean(v) for k, v in acc.items()}
        
        metrics['Accuracy_overall'] = np.mean(all_acc)
        print(metrics)
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

from torch.utils.data import Dataset
import os
import numpy as np
class MVBench_dataset(Dataset):
    def __init__(self, data_dir, split, num_frames=8):
        self.data_list = load_dataset(data_dir,split)['train']
        self.prefix = 'dataset/mvbench/video/'+data_list[split][1].replace('your_data_path/','')
        self.decord_method = {
            'video': self.read_video,
            'gif': self.read_gif,
            'frame': self.read_frame,
        }
        self.split = split
        self.sub_video = data_list[split][3]
        self.use_frame = data_list[split][2] == 'frame'
        self.num_segments = num_frames

    def __str__(self):
        len_list = {}
        option_list = {}
        for data in self.data_list:
            if data['task_type'] not in len_list:
                len_list[data['task_type']] = 0
            len_list[data['task_type']] += 1
            if data['task_type'] not in option_list:
                option_list[data['task_type']] = 0
            option_list[data['task_type']] += len(data['data']['candidates'])
        
        correct = 0
        total = 0
        res = f"There are {len(self.data_list)} videos as follow:\n"
        for k, v in len_list.items():
            correct += len_list[k]
            total += option_list[k]
            res += f"{v} for {k} ({option_list[k]} options => {len_list[k]/option_list[k]*100:.2f}%)\n"
            correct = correct + 1 / option_list[k]
        res += f"Total random accuracy: {correct/total*100:.2f}%"
        return res.rstrip()
        
    def __len__(self):
        return len(self.data_list)
    
    def get_index(self, bound, fps, max_frame, first_idx=0):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / self.num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(self.num_segments)
        ])
        return frame_indices
    
    def read_video(self, video_path, bound=None):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0) 
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].numpy())
            images_group.append(img)
        torch_imgs = self.transform(images_group)
        return torch_imgs
    
    def read_gif(self, video_path, bound=None, fps=25):
        gif = imageio.get_reader(video_path)
        max_frame = len(gif) - 1
        
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0) 
        for index, frame in enumerate(gif):
            if index in frame_indices:
                img = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                img = Image.fromarray(img)
                images_group.append(img)
        torch_imgs = self.transform(images_group)
        return torch_imgs
    
    def read_frame(self, video_path, bound=None, fps=3):
        max_frame = len(os.listdir(video_path))
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=1) # frame_idx starts from 1
        for frame_index in frame_indices:
            img = Image.open(os.path.join(video_path, f"{frame_index:05d}.jpg"))
            images_group.append(img)
        torch_imgs = self.transform(images_group)
        return torch_imgs

    def qa_template(self, data):
        question = f"Question: {data['question']}\n"
        question += "Options:\n"
        answer = data['answer']
        answer_idx = -1
        for idx, c in enumerate(data['candidates']):
            question += f"{chr(ord('A') + idx)}. {c}\n"
            if c == answer:
                answer_idx = idx
        question = question.rstrip()
        answer = f"{chr(ord('A') + answer_idx)}. {answer}"
        return question, answer

    def __getitem__(self, idx):
        if self.sub_video:
            bound = (
                self.data_list[idx]['start'],
                self.data_list[idx]['end'],
            )
        else:
            bound = None
            
        if self.use_frame:
            image = []
            video_path = os.path.join(self.prefix,self.data_list[idx]['video'])
            max_frame = len(os.listdir(video_path))
            images_group = list()
            fps = 3
            frame_indices = self.get_index(bound, fps, max_frame, first_idx=1) # frame_idx starts from 1
            for frame_index in frame_indices:
                img = os.path.join(video_path, f"{frame_index:05d}.jpg")
                image.append(img)
            video_path = None
        else:
            video_path = os.path.join(self.prefix,self.data_list[idx]['video'])
            image = None

        question, answer = self.qa_template(self.data_list[idx])
            
        return {
            'video': video_path,
            'image': image,
            'question': question, 
            'answer': answer,
            'bound': bound,
            'category': self.split
        }
