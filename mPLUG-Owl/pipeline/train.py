import argparse
from functools import partial

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from sconf import Config
from icecream import ic
from peft import LoraConfig, get_peft_config, get_peft_model
from transformers import Trainer
from transformers.training_args import TrainingArguments

from mplug_owl import MplugOwlForConditionalGeneration, MplugOwlTokenizer
from pipeline.data_utils import train_valid_test_datasets_provider
from pipeline.utils import batchify, set_args


parser = argparse.ArgumentParser()
# Model
parser.add_argument('--pretrained-ckpt', type=str, default='MAGAer13/mplug-owl-llama-7b-pt',
                    help='Path to the pretrained checkpoint.')
parser.add_argument('--inference_mode', type=bool, default=False,
                    help='The inference mode.')
parser.add_argument('--seq-length', type=int, default=1024,
                    help='Maximum sequence length to process.')
parser.add_argument('--use-lora', action='store_true', help='LORA.')
parser.add_argument('--lora-r', type=int, default=8,
                    help='curvature.')
parser.add_argument('--lora-alpha', type=int, default=32,
                    help='The initialization coefficient of lora-alpha.')  
parser.add_argument('--lora-dropout', type=int, default=0.05,
                    help='The initialization coefficient of lora_dropout.')
parser.add_argument('--bf16', action='store_true', default=True,
                    help='Run model in bfloat16 mode.')

# Data
parser.add_argument('--mm-config', type=str, default=None, help='Multimodal Config.')
parser.add_argument('--num-workers', type=int, default=8,
                    help="Dataloader number of workers.")  

# Training HyperParameters
parser.add_argument('--train-epochs', type=int, default=3,
                    help='Total number of epochs to train over all '
                    'training runs.')
parser.add_argument('--micro-batch-size', type=int, default=None,
                    help='Batch size per model instance (local batch size). '
                    'Global batch size is local batch size times data '
                    'parallel size times number of micro batches.')
parser.add_argument('--lr', type=float, default=None,
                    help='Initial learning rate. Depending on decay style '
                    'and initial warmup, the learing rate at each '
                    'iteration would be different.')
parser.add_argument('--min-lr', type=float, default=1e-6,
                    help='Minumum value for learning rate. The scheduler'
                    'clip values below this threshold.')
parser.add_argument('--weight-decay', type=float, default=0.01,
                    help='Weight decay coefficient for L2 regularization.')
parser.add_argument('--gradient-accumulation-steps', type=int, default=8,
                    help='The gradient accumulation steps.')
parser.add_argument('--clip-grad', type=float, default=1.0,
                    help='Gradient clipping based on global L2 norm.')
parser.add_argument('--adam-beta1', type=float, default=0.9,
                    help='First coefficient for computing running averages '
                    'of gradient and its square')
parser.add_argument('--adam-beta2', type=float, default=0.999,
                    help='Second coefficient for computing running averages '
                    'of gradient and its square')
parser.add_argument('--adam-eps', type=float, default=1e-08,
                    help='Term added to the denominator to improve'
                    'numerical stability')

parser.add_argument('--num-warmup-steps', type=int, default=50,
                    help='The number of warmup steps.')
parser.add_argument('--num-training-steps', type=int, default=4236,
                    help='The number of total training steps for lr scheduler.')

# Evaluation & Save
parser.add_argument('--save-path', type=str, default=None,
                    help='Output directory to save checkpoints to.')
parser.add_argument('--save-interval', type=int, default=None,
                    help='Number of iterations between checkpoint saves.')
parser.add_argument('--eval-iters', type=int, default=100,
                    help='Number of iterations to run for evaluation'
                    'validation/test for.')

# Other
parser.add_argument('--gradient-checkpointing', action='store_true',
                    help='The gradient checkpointing.')
parser.add_argument('--logging-nan-inf-filter', action='store_true',
                    help='The logging nan inf filter.')
parser.add_argument('--ddp-find-unused-parameters', action='store_true',
                    help='unused parameters finding.')
parser.add_argument('--do-train', action='store_true', default=True,
                    help='Whether to do training.')  
parser.add_argument('--local_rank', type=int, default=-1,
                    help='Local rank')



class CustomTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def get_train_dataloader(self) -> DataLoader:
        dataset = self.train_dataset
        sampler = DistributedSampler(dataset)
        return torch.utils.data.DataLoader(
            dataset, batch_size=self._train_batch_size,
            sampler=sampler,
            num_workers=self.args.dataloader_num_workers,
            drop_last=True,
            pin_memory=True,
            collate_fn=batchify)


    def get_eval_dataloader(self, eval_dataset: Dataset | None = None) -> DataLoader:
        dataset = self.eval_dataset
        sampler = DistributedSampler(dataset, shuffle=False)
        return torch.utils.data.DataLoader(
            dataset, batch_size=self._train_batch_size,
            sampler=sampler,
            num_workers=self.args.dataloader_num_workers,
            drop_last=True,
            pin_memory=True,
            collate_fn=batchify)


def main():
    args, left_argv = parser.parse_known_args()  
    ic(left_argv)
    config = Config(args.mm_config)

    set_args(args)

    model = MplugOwlForConditionalGeneration.from_pretrained(
        args.pretrained_ckpt,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.half,
    )
    tokenizer = MplugOwlTokenizer.from_pretrained(args.pretrained_ckpt)

    if args.use_lora:
        for param in model.parameters():
            # freeze base model's layers
            param.requires_grad = False
        peft_config = LoraConfig(
            target_modules=r'.*language_model.*\.(q_proj|v_proj)', 
            inference_mode=args.inference_mode, 
            r=args.lora_r, 
            lora_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        if args.gradient_checkpointing:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.language_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
            model.language_model.apply(
                partial(model.language_model._set_gradient_checkpointing, value=True))

    else:
        for name, param in model.named_parameters():
            if 'language_model' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        if args.gradient_checkpointing:
            model.language_model.apply(
                partial(model.language_model._set_gradient_checkpointing, value=True))

    model.train()

    train_data, valid_data = train_valid_test_datasets_provider(
        config.data_files, config=config, 
        tokenizer=tokenizer, seq_length=args.seq_length
    )

    trainer = CustomTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=TrainingArguments(
            learning_rate=args.lr,
            warmup_steps=args.num_warmup_steps,
            do_train=args.do_train,
            num_train_epochs=args.train_epochs,
            output_dir=args.save_path,
            save_strategy='steps',
            save_steps=args.save_interval,
            evaluation_strategy='steps',
            eval_steps=args.eval_iters,
            per_device_train_batch_size=args.micro_batch_size,
            max_grad_norm=args.clip_grad,
            weight_decay=args.weight_decay,
            bf16=args.bf16,
            fp16=not args.bf16,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_checkpointing=args.gradient_checkpointing,
            logging_steps=args.eval_iters//4,
            logging_nan_inf_filter=args.logging_nan_inf_filter,
            ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        ),
    )

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train()

    model.save_pretrained(args.save_path)

if __name__ == '__main__':
    main()