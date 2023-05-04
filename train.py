
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

    def _wrap_model(self, model, training=True, dataloader=None):
        if self.args.use_ipex:
            dtype = torch.bfloat16 if self.use_cpu_amp else torch.float32
            model = self.ipex_optimize_model(model, training, dtype=dtype)

        if self.args.jit_mode_eval:
            model = self.torch_jit_model_eval(model, dataloader, training)

        if is_sagemaker_mp_enabled():
            # Wrapping the base model twice in a DistributedModel will raise an error.
            if isinstance(self.model_wrapped, smp.model.DistributedModel):
                return self.model_wrapped
            return smp.DistributedModel(model, backward_passes_per_step=self.args.gradient_accumulation_steps)

        # already initialized its own DDP and AMP
        if self.deepspeed:
            return self.deepspeed

        # train/eval could be run multiple-times - if already wrapped, don't re-wrap it again
        if unwrap_model(model) is not model:
            return model

        # Mixed precision training with apex (torch < 1.6)
        if self.use_apex and training:
            model, self.optimizer = amp.initialize(
                model, self.optimizer, opt_level=self.args.fp16_opt_level)

        # Multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = nn.DataParallel(model)

        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.
        if not training:
            return model

        # Distributed training (should be after apex fp16 initialization)
        if self.sharded_ddp is not None:
            from fairscale.nn.data_parallel import FullyShardedDataParallel as FullyShardedDDP
            from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
            from fairscale.nn.wrap import auto_wrap
            from fairscale.optim import OSS
            from fairscale.optim.grad_scaler import ShardedGradScaler
            # Sharded DDP!
            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                model = ShardedDDP(model, self.optimizer)
            else:
                mixed_precision = self.args.fp16 or self.args.bf16
                cpu_offload = ShardedDDPOption.OFFLOAD in self.args.sharded_ddp
                zero_3 = self.sharded_ddp == ShardedDDPOption.ZERO_DP_3
                # XXX: Breaking the self.model convention but I see no way around it for now.
                if ShardedDDPOption.AUTO_WRAP in self.args.sharded_ddp:
                    model = auto_wrap(model)
                self.model = model = FullyShardedDDP(
                    model,
                    mixed_precision=mixed_precision,
                    reshard_after_forward=zero_3,
                    cpu_offload=cpu_offload,
                ).to(self.args.device)

        # Distributed training using PyTorch FSDP
        if self.fsdp is not None:
            # PyTorch FSDP!
            from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
            from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision
            from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy

            if FSDPOption.OFFLOAD in self.args.fsdp:
                cpu_offload = CPUOffload(offload_params=True)
            else:
                cpu_offload = CPUOffload(offload_params=False)
            import functools
            auto_wrap_policy = None
            if FSDPOption.AUTO_WRAP in self.args.fsdp:
                if self.args.fsdp_min_num_params > 0:
                    auto_wrap_policy = functools.partial(
                        size_based_auto_wrap_policy, min_num_params=self.args.fsdp_min_num_params
                    )
                elif self.args.fsdp_transformer_layer_cls_to_wrap is not None:
                    transformer_cls_to_wrap = get_module_class_from_name(
                        model, self.args.fsdp_transformer_layer_cls_to_wrap
                    )
                    auto_wrap_policy = functools.partial(
                        transformer_auto_wrap_policy,
                        # Transformer layer class to wrap
                        transformer_layer_cls={transformer_cls_to_wrap},
                    )
            mixed_precision_policy = None
            dtype = None
            if self.args.fp16:
                dtype = torch.float16
            elif self.args.bf16:
                dtype = torch.bfloat16
            if dtype is not None:
                mixed_precision_policy = MixedPrecision(
                    param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)
            if type(model) != FSDP:
                # XXX: Breaking the self.model convention but I see no way around it for now.
                self.model = model = FSDP(
                    model,
                    sharding_strategy=self.fsdp,
                    cpu_offload=cpu_offload,
                    auto_wrap_policy=auto_wrap_policy,
                    mixed_precision=mixed_precision_policy,
                )
                if FSDPOption.OFFLOAD not in self.args.fsdp:
                    model.to(self.args.device)

        elif is_sagemaker_dp_enabled():
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[int(os.getenv("SMDATAPARALLEL_LOCAL_RANK"))]
            )
        elif self.args.local_rank != -1:
            kwargs = {}
            if self.args.ddp_find_unused_parameters is not None:
                kwargs["find_unused_parameters"] = self.args.ddp_find_unused_parameters
            elif isinstance(model, PreTrainedModel):
                # find_unused_parameters breaks checkpointing as per
                # https://github.com/huggingface/transformers/pull/4659#issuecomment-643356021
                kwargs["find_unused_parameters"] = not model.is_gradient_checkpointing
            else:
                kwargs["find_unused_parameters"] = True

            if self.args.ddp_bucket_cap_mb is not None:
                kwargs["bucket_cap_mb"] = self.args.ddp_bucket_cap_mb
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[
                    self.args.local_rank] if self.args._n_gpu != 0 else None,
                output_device=self.args.local_rank if self.args._n_gpu != 0 else None,
                static_graph=False,
                **kwargs,
            )

        return model


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

    from interface import get_model, get_model_toy

    # model, tokenizer, img_processor = get_model_toy(tokenizer_path=args.vocab_file)
    model, tokenizer, img_processor = get_model(
        checkpoint_path='pretrained.pth', tokenizer_path=args.vocab_file)

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
            if 'vision_model' not in name:
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
