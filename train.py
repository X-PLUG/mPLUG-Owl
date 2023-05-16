
import argparse
from functools import partial
import math
import os
from pathlib import Path
from typing import (TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple,
                    Union)

import torch
from icecream import ic
from peft import LoraConfig, TaskType, get_peft_config, get_peft_model
from sconf import Config
from torch import nn
from torch.utils.data import (DataLoader, Dataset, RandomSampler,
                              SequentialSampler)
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from transformers import Trainer
from transformers.data.data_collator import DataCollator
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers.training_args import TrainingArguments

from data_utils import train_valid_test_datasets_provider
from data_utils.processors import *
from utils import (batchify, get_args, get_cosine_schedule_with_warmup, get_param_groups, print_rank_0, set_args, set_tokenizer,
                   worker_init)
from transformers.utils import is_sagemaker_mp_enabled, is_sagemaker_dp_enabled
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from transformers.trainer_utils import (
    FSDPOption,
    ShardedDDPOption,
)
from transformers.trainer_pt_utils import (get_module_class_from_name,)


class CustomTrainer(Trainer):
    def __init__(self, data_path, model: PreTrainedModel | nn.Module = None, args: TrainingArguments = None, data_collator: Any | None = None, train_dataset: Dataset | None = None, eval_dataset: Dataset | None = None, tokenizer: PreTrainedTokenizerBase | None = None, model_init: Callable[[], PreTrainedModel] = None, compute_metrics: Callable[[EvalPrediction], Dict] | None = None, callbacks: List[TrainerCallback] | None = None, optimizers=..., preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None):
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer,
                         model_init, compute_metrics, callbacks, optimizers, preprocess_logits_for_metrics)

        train_ds, valid_ds, test_ds = train_valid_test_datasets_provider(
            data_path, iters_per_epoch=None, config=config)
        self.train_ds = train_ds
        self.valid_ds = valid_ds

    def get_train_dataloader(self) -> DataLoader:
        dataset = self.train_ds
        from torch.utils import data
        sampler = data.DistributedSampler(dataset)

        worker_init_obj = worker_init(
            0 if self.state.epoch is None else self.state.epoch)
        return torch.utils.data.DataLoader(dataset,
                                           batch_size=args.micro_batch_size,
                                           sampler=sampler,
                                           num_workers=args.num_workers,
                                           drop_last=True,
                                           pin_memory=True,
                                           collate_fn=batchify,
                                           prefetch_factor=4,
                                           worker_init_fn=worker_init_obj._worker_init_fn)

    def get_eval_dataloader(self, eval_dataset: Dataset | None = None) -> DataLoader:
        dataset = self.valid_ds
        from torch.utils import data
        sampler = data.DistributedSampler(dataset)

        worker_init_obj = worker_init(self.state.epoch)
        return torch.utils.data.DataLoader(dataset,
                                           batch_size=args.micro_batch_size,
                                           sampler=sampler,
                                           num_workers=args.num_workers,
                                           drop_last=True,
                                           pin_memory=True,
                                           collate_fn=batchify,
                                           prefetch_factor=4,
                                           worker_init_fn=worker_init_obj._worker_init_fn)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    from utils import (_add_checkpointing_args, _add_data_args,
                       _add_learning_rate_args, _add_mixed_precision_args,
                       _add_multimodal_args, _add_regularization_args,
                       _add_training_args, _add_validation_args)
    parser = _add_regularization_args(parser)
    parser = _add_training_args(parser)
    parser = _add_validation_args(parser)
    parser = _add_learning_rate_args(parser)
    parser = _add_mixed_precision_args(parser)
    parser = _add_data_args(parser)
    parser = _add_multimodal_args(parser)
    parser = _add_checkpointing_args(parser)
    args, left_argv = parser.parse_known_args()
    ic(left_argv)
    config = Config(args.mm_config)

    set_args(args)

    from interface import get_model

    model, tokenizer, img_processor = get_model(
        checkpoint_path='pretrained.pth', tokenizer_path=args.vocab_file, device='cpu')

    if args.use_lora:
        for param in model.parameters():
            # freeze base model's layers
            param.requires_grad = False
        peft_config = LoraConfig(
            target_modules=r'.*language_model.*\.(q_proj|v_proj)', inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.05
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    else:
        for name, param in model.named_parameters():
            if 'language_model' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        model.language_model.apply(
            partial(model.language_model._set_gradient_checkpointing, value=True))
    if args.bf16:
        model = model.bfloat16()
    else:
        model = model.float()
    model.train()
    set_tokenizer(tokenizer)
    from torch.optim import AdamW
    param_groups = get_param_groups(
        [model], no_weight_decay_cond=None, scale_lr_cond=lambda x, y: 'vision_model' in x, lr_mult=0.1)
    optimizer = AdamW(param_groups,
                      lr=args.lr,
                      weight_decay=args.weight_decay,
                      betas=(args.adam_beta1, args.adam_beta2),
                      eps=args.adam_eps)
    # lr_scheduler = get_optimizer_param_scheduler(optimizer)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, lr=args.lr, min_lr=args.min_lr, num_warmup_steps=50, num_training_steps=4236)

    trainer = CustomTrainer(
        data_path=config.data_files,
        model=model,
        optimizers=(optimizer, lr_scheduler),
        # compute_metrics=compute_metrics,
        args=TrainingArguments(
            do_train=True,
            num_train_epochs=args.train_epochs,
            output_dir=args.save,
            save_strategy='steps',
            save_steps=args.save_interval,
            evaluation_strategy='steps',
            eval_steps=args.eval_iters,
            per_device_train_batch_size=args.micro_batch_size,  # 4, global = 4*64
            max_grad_norm=args.clip_grad,
            weight_decay=args.weight_decay,
            bf16=args.bf16,
            gradient_accumulation_steps=8,
            gradient_checkpointing=False,
            logging_steps=args.eval_iters//4,
            logging_nan_inf_filter=False,
            ddp_find_unused_parameters=False,
        ),
    )

    # for batch in trainer.get_train_dataloader():
    #     print(batch)
    #     input()
    trainer.train()
