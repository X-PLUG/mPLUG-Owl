"""Evaluate GPT"""
import copy
import PIL

from collections import defaultdict
import torch
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import ruamel.yaml as yaml

from functools import partial

from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoProcessor
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data.sampler import Sampler

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os
import json
import time
import itertools
from tqdm import tqdm
import os
import argparse

from tasks.task_define.base import ResulterSaverBase, StandardData, build_evaluation_from_jsonl, build_evaluation_from_yaml
from icecream import ic
from datetime import timedelta

import pdb


def initialize_distributed(args):
    """Initialize torch.distributed and core model parallel."""

    device_count = torch.cuda.device_count()
    
    if args.rank == 0:
        print('> initializing torch distributed ...', flush=True)
    # Manually set the device ids.
    if device_count > 0:
        device = args.rank % device_count
        if args.local_rank is not None:
            assert args.local_rank == device, \
                'expected local-rank to be the same as rank % device-count.'
        else:
            args.local_rank = device
        torch.cuda.set_device(device)
    # Call the init process
    torch.distributed.init_process_group(
        backend='nccl',
        world_size=args.world_size, rank=args.rank,
        timeout=timedelta(minutes=600))


class InferenceSamplerV2(Sampler):

    def __init__(self, size):
        if isinstance(size, float):
            logger.info(f"InferenceSampler(size=) expects an int but gets float, convert from {size} to {int(size)}.")
            size = int(size)
        elif not isinstance(size, int):
            raise TypeError(f"InferenceSampler(size=) expects an int. Got type {type(size)}.")
        self._size = size
        assert size > 0
        self._rank = dist.get_rank()
        self._world_size = dist.get_world_size()
        self._local_indices = [i for i in range(size) if i%self._world_size==self._rank]

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[: rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, task):
        inference_processor_config = {
            'default': {
                'type': 'BaseSFTProcessorInference',
                'image_size': 448,
                'min_scale': 1.0,
                'randaug': False,
                'prompt_style': 'vicuna',
                'enable_system_prompt': False,
                'force_resize': True
            }
        }
        cut_cfg = task.cut_cfg
        if cut_cfg:
            self.cut_enable = True
        else:
            self.cut_enable = False
        self.task = task
        
    def __len__(self):
        return len(self.task)

    def __getitem__(self, idx):
        task_data: StandardData = self.task[idx]
        messages, images, videos, gt_answer, index = task_data.messages, task_data.images, task_data.videos, task_data.gt_answer, task_data.index
        images = [x.convert('RGB') if isinstance(x, PIL.Image.Image) else self.task.io(x, auto_retry=True)  for x in images]
        if len(videos)>0:
            assert len(images) == 0
            video_urls = videos
            batch_frames = [self.task.io._load_video(_)[0] for _ in video_urls]
            num_frame = set(len(_) for _ in batch_frames)
            assert len(num_frame) == 1
            num_frame = list(num_frame)[0]
            images = [_ for frames in batch_frames for _ in frames]
            for msg in messages:
                msg['content']=msg['content'].replace('<|video|>','<|image|>'*num_frame)

        assert messages[-1]['role'] == 'assistant'
        return images, gt_answer, index, task_data, messages, self.cut_enable
    
def collate_fn(batches):
    return batches

def get_args():
    parser = argparse.ArgumentParser(description="Evaluation")

    parser.add_argument('--evaluation-plan', type=str, help="a json or yaml path to evaluation plan", required=True)
    parser.add_argument('--load', type=str, help="path to the checkpoint", required=True)
    parser.add_argument('--num_workers', type=int, default=8)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()

    args.world_size = int(os.environ['WORLD_SIZE'])
    args.rank = int(os.environ['LOCAL_RANK'])
    args.local_rank = args.rank # single machine

    initialize_distributed(args)
    
    config = AutoConfig.from_pretrained(args.load, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.load, attn_implementation='sdpa', torch_dtype=torch.half, trust_remote_code=True)

    device = torch.device(f'cuda:{args.rank}')
    model.eval().to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(args.load)
    processor = model.init_processor(tokenizer)
    
    if args.evaluation_plan.endswith('.jsonl'):
        tasks = build_evaluation_from_jsonl(args.evaluation_plan)
    elif args.evaluation_plan.endswith('.yaml'):
        tasks = build_evaluation_from_yaml(args.evaluation_plan)
    else:
        raise NotImplementedError
    
    for task_i, task in enumerate(tasks):

        if dist.get_rank() == 0:
            saver = ResulterSaverBase(task=task, checkpoint_path=args.load)
        dataset = InferenceDataset(
            task=task,
        )
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=InferenceSamplerV2(size=len(dataset)),
            batch_size=1, # we do not support batchsize>1
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )
        outputs = []
        for _, batch in enumerate(tqdm(loader, desc=f'{task_i+1}/{len(tasks)}: '+task.ds_name+f'[RANK {args.local_rank}]')):
            images, gt_answer, index, task_data, messages, cut_enable = batch[0]

            inputs = processor(messages, images=images, videos=None, cut_enable=cut_enable)
            inputs.to('cuda')
            inputs.update({
                'tokenizer': tokenizer,
                'max_new_tokens': 100,
                'decode_text': True,
                'do_sample': True,
                'top_k': 1,
            })

            with torch.no_grad():
                answers = model.generate(**inputs)
            task_data.raw_model_answer = answers[0]
            outputs.append(task.postprocess(task_data))

        print(task.ds_name+f'[RANK {args.local_rank}] Processed.')
        torch.distributed.barrier()
        world_size = dist.get_world_size()
        merged_outputs = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_outputs, outputs)

        merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

        if dist.get_rank() == 0:
            
            results_file = saver.save_output_results(merged_outputs)
            submission_file = task.build_submission(merged_outputs, saver)
            metrics, _ = task.evaluate(merged_outputs, results_file, submission_file)
            saver.save_metrics(metrics)
            
            
        torch.distributed.barrier()
