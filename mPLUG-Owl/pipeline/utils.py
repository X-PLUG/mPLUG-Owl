import math
import random
import torch
import numpy as np
from icecream import ic

def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)

ARGS = None
def set_args(args):
    global ARGS
    ARGS = args

def get_args():
    return ARGS

TOKENIZER = None
def set_tokenizer(tokenizer):
    global TOKENIZER
    TOKENIZER = tokenizer

def get_tokenizer():
    return TOKENIZER
from torch import distributed as dist

class worker_init:
    def __init__(self, epoch_id):
        self.epoch_id = epoch_id
    def _worker_init_fn(self, worker_id):
        random.seed(worker_id + self.epoch_id*1e4 + dist.get_rank()*1e8)


def batchify(batch):
    # collate_fn
    # image = torch.cat([data["image"] for data in batch], dim=0)
    image = [data["image"] if data["image"] is not None else None for data in batch]
    if all([img is None for img in image]):
        image = None
    else:
        image = torch.cat([img for img in image if img is not None], dim=0)
    num_images_per_sample = torch.LongTensor([data["image"].size(0) if data['image'] is not None else 0 for data in batch])

    text = torch.stack([torch.LongTensor(data["text"]['input_ids']) for data in batch], dim=0)
    non_padding_mask = torch.stack([torch.LongTensor(data["text"]['non_padding_mask']) for data in batch], dim=0)
    non_media_mask = torch.stack([torch.LongTensor(data["text"]['non_media_mask']) for data in batch], dim=0)
    prompt_mask = torch.stack([torch.LongTensor(data["text"]['prompt_mask']) for data in batch], dim=0)
    prompt_length = torch.from_numpy(np.stack([data["text"]["prompt_length"] for data in batch]))
    seq_length = torch.from_numpy(np.stack([data["text"]["seq_length"] for data in batch]))
    
    output_batch = {
        "pixel_values": image,
        "input_ids": text.long(),
        "labels": text.long().clone(),
        "num_images": num_images_per_sample.long(),
        "non_padding_mask": non_padding_mask.long(),
        "non_media_mask": non_media_mask.long(),
        "prompt_mask": prompt_mask.long()        
    }
    return output_batch


def get_param_groups(modules,
                     no_weight_decay_cond,
                     scale_lr_cond,
                     lr_mult):
    """creates param groups based on weight decay condition (regularized vs non regularized)
       and learning rate scale condition (args.lr vs lr_mult * args.lr)
       scale_lr_cond is used during finetuning where head of the network requires a scaled
       version of the base learning rate. 
    """
    wd_no_scale_lr = []
    wd_scale_lr = []
    no_wd_no_scale_lr = []
    no_wd_scale_lr = []
    for module in modules:
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue

            if no_weight_decay_cond is not None:
                no_wd = no_weight_decay_cond(name, param)
            else:
                # do not regularize biases nor Norm parameters
                no_wd = name.endswith(".bias") or len(param.shape) == 1

            if scale_lr_cond is not None:
                scale_lr = scale_lr_cond(name, param)
            else:
                scale_lr = False

            if not no_wd and not scale_lr:
                wd_no_scale_lr.append(param)
            elif not no_wd and scale_lr:
                wd_scale_lr.append(param)
            elif no_wd and not scale_lr:
                no_wd_no_scale_lr.append(param)
            else:
                no_wd_scale_lr.append(param)

    param_groups = []
    if len(wd_no_scale_lr):
        param_groups.append(
            {'params': wd_no_scale_lr, 'wd_mult': 1.0, 'lr_mult': 1.0})
    if len(wd_scale_lr):
        param_groups.append(
            {'params': wd_scale_lr, 'wd_mult': 1.0, 'lr_mult': lr_mult})
    if len(no_wd_no_scale_lr):
        param_groups.append({'params': no_wd_no_scale_lr,
                            'wd_mult': 0.0, 'lr_mult': 1.0})
    if len(no_wd_scale_lr):
        param_groups.append(
            {'params': no_wd_scale_lr, 'wd_mult': 0.0, 'lr_mult': lr_mult})

    return param_groups

def get_cosine_schedule_with_warmup(
        optimizer, lr, min_lr, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
    ):
        """
        Create a schedule with a learning rate that decreases following the values of the cosine function between the
        initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
        initial lr set in the optimizer.

        Args:
            optimizer ([`~torch.optim.Optimizer`]):
                The optimizer for which to schedule the learning rate.
            num_warmup_steps (`int`):
                The number of steps for the warmup phase.
            num_training_steps (`int`):
                The total number of training steps.
            num_cycles (`float`, *optional*, defaults to 0.5):
                The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
                following a half-cosine).
            last_epoch (`int`, *optional*, defaults to -1):
                The index of the last epoch when resuming training.

        Return:
            `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
        """

        delta_min_lr = (lr-min_lr)/lr  # 0.95

        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return (1-delta_min_lr) + delta_min_lr * float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / \
                float(max(1, num_training_steps - num_warmup_steps))
            return delta_min_lr + (1-delta_min_lr) * max(0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
        from torch.optim.lr_scheduler import LambdaLR
        return LambdaLR(optimizer, lr_lambda, last_epoch)