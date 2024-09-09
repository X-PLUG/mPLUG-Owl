import json
import os
import time
from typing import Dict, List
from .registry import Registry, build_from_cfg
import yaml
from dataclasses import dataclass
from pathlib import Path
from x.io import write_json
from dataclasses import field
from icecream import ic
TASK = Registry('task')

from PIL import Image
import traceback

from .video_utils import read_frames_decord
from torchvision import transforms


@dataclass()
class StandardData:
    messages: list = field(default_factory=list)
    images: list = field(default_factory=list)
    videos: list = field(default_factory=list)
    gt_answer: str = ''
    index: int = 0
    extra: dict = field(default_factory=dict) # for evaluation
    raw_model_answer = None
    model_answer = None

    def to_json(self,):
        return {
            'messages': self.messages,
            'images': self.images,
            'videos': self.videos,
            'gt_answer': self.gt_answer,
            'index': self.index,
            'extra': self.extra,
            'raw_model_answer': self.raw_model_answer,
            'model_answer': self.model_answer
        }


def empty_image():
    return Image.new('RGB', (800, 600), (255, 255, 255))


class ImageIO(object):
    def __init__(self):
        self.retry_num = 10

    def __call__(self, image_url, auto_retry=False, raise_error=False):
        for i in range(self.retry_num):
            try:
                if os.path.isfile(image_url):
                    image = Image.open(image_url).convert('RGB')
                return image
            except Exception as e:
                traceback.print_exc()
                if auto_retry:
                    pass
                else:
                    if raise_error:
                        raise RuntimeError(image_url)
                    ic()
                    return empty_image()

    def _load_video(self, video_url, num_frames=8):
        video_tensors = []
        timestamps = []

        if isinstance(video_url, dict):
            if 'bound' in video_url:
                start_time = video_url['bound'][0]
                end_time = video_url['bound'][1]
            else:
                start_time = None
                end_time = None
            num_frames = video_url.get('num_frames', num_frames)
            video_url = video_url['video']
        else:
            start_time = None
            end_time = None
            video_url = str(video_url)

        video, timestamp = read_frames_decord(video_url, num_frames=num_frames, sample='middle', start_time=start_time, end_time=end_time)
        
        to_pil = transforms.ToPILImage()
        frames = [to_pil(video[ti]) for ti in range(video.shape[0])]

        return frames, timestamp


@dataclass()
class ModelInfo():
    evaluate_model: str = ''
    evaluate_model_setting: str = ''


@TASK.register_module()
class EvalTaskBase():
    def __init__(self, task_config, generate_config={}, cut_cfg=None) -> None:
        self.task_config = task_config
        self.cut_cfg = cut_cfg
        self.io = ImageIO()
        self.batch_size = task_config.get('batch_size', None)
        self.enable_system_prompt = task_config.get('enable_system_prompt',False)
        self.seed = 0
        self.generate_config = generate_config
        ic(self.generate_config)
        if self.enable_system_prompt:
            if 'system_prompt' in task_config:
                self.system_prompt = task_config['system_prompt']
            else:
                self.system_prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. "
    
    def __getitem__(self, index) -> StandardData:
        raise NotImplementedError

    def postprocess(self, line: StandardData):
        return line.to_json()

    def build_submission(self, merged_lines: List[Dict], saver):
        return None

    def evaluate(self, merged_outputs, results_file=None, submission_file=None):
        pass

def easy_name(parent_dir, child_name):
    prefix_number = 0
    parent_dir = Path(parent_dir)
    parent_dir.mkdir(exist_ok=True, parents=True)
    child_map = {}
    curr_max_num = -1
    for subfile in parent_dir.iterdir():
        if subfile.name.split('.')[0].isdigit():
            child_map['.'.join(subfile.name.split('.')[1:])]=subfile.name
            curr_max_num = max(curr_max_num, int(subfile.name.split('.')[0]))
        else:
            child_map[subfile.name]=subfile.name

    if child_name in child_map:
        return child_map[child_name]
    return f'{curr_max_num+1}.{child_name}'

def create_soft_link(src, dst):
    if not os.path.exists(dst):
        os.symlink(src, dst)
        print(f'Symbolic link created: {src} -> {dst}')
    else:
        print(f'The link {dst} already exists.')

class ResulterSaverBase():
    def __init__(self, task, result_dir='output_eval_results_v4',checkpoint_path='default') -> None:
        self.task = task
        self.checkpoint_path = checkpoint_path

        self.model_suffix = str(Path(checkpoint_path).absolute()).split('/')[-2]
        self.iter_name = str(Path(checkpoint_path).absolute()).split('/')[-1]

        self.result_dir = result_dir
        self.time_prefix = time.strftime("%y%m%d%H%M%S", time.localtime())

        # self.results_file = f'{self.get_save_dir()}_{self.time_prefix}_output.json'
        # self.metrics_file = f'{self.get_save_dir()}_{self.time_prefix}_metrics.json'

        # self.results_file = str(Path(self.get_save_dir(), f'{self.task.ds_name}_{self.time_prefix}_output.json').absolute())
        # self.metrics_file = str(Path(self.get_save_dir(), f'{self.task.ds_name}_{self.time_prefix}_metrics.json').absolute())

        self.results_file = f'{self.task.ds_name}_{self.time_prefix}_output.json'
        self.metrics_file = f'{self.task.ds_name}_{self.time_prefix}_metrics.json'
        

    def get_save_dir(self):
        model_suffix_indexed = easy_name(parent_dir=self.result_dir, child_name=self.model_suffix)
        test_prefix_dir = Path(self.result_dir, model_suffix_indexed, self.iter_name)
        if not os.path.exists(test_prefix_dir):
            os.makedirs(test_prefix_dir, exist_ok=True)
        return test_prefix_dir
        # save_dir = str(Path(test_prefix_dir, self.task.ds_name).absolute())
        # return save_dir

    def save_output_results(self, results_after_post_process: List[Dict], ):
        results_file = Path(self.checkpoint_path,'evaluation',self.results_file)
        write_json(results_after_post_process, results_file)
        create_soft_link(results_file, Path(self.get_save_dir(), self.results_file))
        return results_file
    
    def save_metrics(self, metrics):
        metrics_file = Path(self.checkpoint_path,'evaluation',self.metrics_file)
        # write_json(metrics, metrics_file)
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent='\t')
        create_soft_link(metrics_file, Path(self.get_save_dir(), self.metrics_file))
        return metrics_file


def build_evaluation_task(task_name, **kwargs):
    kwargs.update({'type': task_name})
    return build_from_cfg(**kwargs)

def build_evaluation_from_jsonl(task_path):
    with open(task_path,'r')as f:
        task_configs = [json.loads(line) for line in f]
    tasks = [build_from_cfg(task_config, TASK) for task_config in task_configs]
    return tasks


def build_evaluation_from_yaml(task_paths):
    tasks = []
    task_paths = task_paths.split(',')
    for task_path in task_paths:
        with open(task_path,'r')as f:
            task_configs = yaml.load(f, Loader=yaml.Loader)
        tasks.extend([build_from_cfg(task_config, TASK) for task_config in task_configs])
    return tasks

def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(line) for line in f]