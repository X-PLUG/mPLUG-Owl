import argparse
import itertools
import json
import os
import random
import time
from functools import partial
from typing import Optional

import torch
from tqdm import tqdm
from PIL import Image

from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mplug_owl2.conversation import conv_templates, SeparatorStyle
from mplug_owl2.model.builder import load_pretrained_model
from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from vqa import VQA
from vqa_eval import VQAEval

ds_collections = {
    'vqav2_val': {
        'train': 'data/vqav2/vqav2_train.jsonl',
        'test': 'data/vqav2/vqav2_val.jsonl',
        'question': 'data/vqav2/v2_OpenEnded_mscoco_val2014_questions.json',
        'annotation': 'data/vqav2/v2_mscoco_val2014_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'vqav2_testdev': {
        'train': 'data/vqav2/vqav2_train.jsonl',
        'test': 'data/vqav2/vqav2_testdev.jsonl',
        'metric': None,
        'max_new_tokens': 10,
    },
    'okvqa_val': {
        'train': 'data/okvqa/okvqa_train.jsonl',
        'test': 'data/okvqa/okvqa_val.jsonl',
        'question': 'data/okvqa/OpenEnded_mscoco_val2014_questions.json',
        'annotation': 'data/okvqa/mscoco_val2014_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'textvqa_val': {
        'train': 'data/textvqa/textvqa_train.jsonl',
        'test': 'data/textvqa/textvqa_val.jsonl',
        'question': 'data/textvqa/textvqa_val_questions.json',
        'annotation': 'data/textvqa/textvqa_val_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
}

def collate_fn(batches, tokenizer):

    questions = [_['question'] for _ in batches]
    question_ids = [_['question_id'] for _ in batches]
    annotations = [_['annotation'] for _ in batches]

    image_tensor = [_['image_tensor'] for _ in batches]

    input_ids = []
    for input_text in questions:
        input_ids.append(tokenizer_image_token(input_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').tolist())
    input_tokens_max_length = max([len(x) for x in input_ids])
    pad_token_id = tokenizer.pad_token_id

    input_ids = [([pad_token_id] * (input_tokens_max_length - len(_)) + _) for _ in input_ids] # pad in the left
    input_ids = torch.LongTensor(input_ids)
    attention_mask = 1 - input_ids.eq(pad_token_id).long()
    
    image_tensor = torch.cat(image_tensor, dim=0)
    return question_ids, image_tensor, input_ids, attention_mask, annotations


class VQADataset(torch.utils.data.Dataset):

    def __init__(self, train, test, prompt, image_processor, few_shot):
        self.test = json.load(open(test))
        self.prompt = prompt
        self.image_processor = image_processor

        self.few_shot = few_shot
        if few_shot > 0:
            self.train = open(train).readlines()

    def __len__(self):
        return len(self.test)

    def __getitem__(self, idx):
        data = self.test[idx]
        image, question, question_id, annotation = data['image'], data[
            'question'], data['question_id'], data.get('answer', None)
        image = Image.open(image).convert('RGB')
        max_edge = max(image.size)
        image = image.resize((max_edge, max_edge)) # Resize here for best performance
        image_tensor = process_images([image], self.image_processor)

        return {
            'image_tensor': image_tensor,
            'question': self.prompt.format(question),
            'question_id': question_id,
            'annotation': annotation
        }


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size,
                                                      self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--dataset', type=str, default='textvqa_val')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--few-shot', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    model_path = args.checkpoint
    model_name = get_model_name_from_path(model_path)
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit=False, load_4bit=False, device_map="cuda", device="cuda")
    tokenizer.padding_side = 'left'
    tokenizer.pad_token_id = tokenizer.eos_token_id

    prompt = 'USER: <|image|>{}\nAnswer the question using a single word or phrase. ASSISTANT:'

    random.seed(args.seed)
    dataset = VQADataset(
        train=ds_collections[args.dataset]['train'],
        test=ds_collections[args.dataset]['test'],
        prompt=prompt,
        image_processor=image_processor,
        few_shot=args.few_shot,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=InferenceSampler(len(dataset)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
    )

    outputs = []
    for _, (question_ids, image_tensor, input_ids, attention_mask,
            annotations) in tqdm(enumerate(dataloader)):
        pred = model.generate(
            input_ids=input_ids.cuda(),
            attention_mask=attention_mask.cuda(),
            images=image_tensor.to(dtype=model.dtype).cuda(),
            do_sample=False,
            num_beams=5,
            max_new_tokens=ds_collections[args.dataset]['max_new_tokens'],
            min_new_tokens=1,
            length_penalty=1,
            num_return_sequences=1,
            output_hidden_states=True,
            use_cache=True,
        )
        answers = [
            tokenizer.decode(_[input_ids.size(1):].cpu(),
                             skip_special_tokens=True).strip() for _ in pred
        ]

        for question_id, answer, annotation in zip(question_ids, answers,
                                                   annotations):
            if args.dataset in ['vqav2_val', 'vqav2_testdev', 'okvqa_val', 'textvqa_val']:
                outputs.append({
                    'question_id': question_id,
                    'answer': answer,
                })
            else:
                raise NotImplementedError

    torch.distributed.barrier()

    world_size = torch.distributed.get_world_size()
    merged_outputs = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))

    merged_outputs = [json.loads(_) for _ in merged_outputs]
    merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

    if torch.distributed.get_rank() == 0:
        print(f"Evaluating {args.dataset} ...")
        time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
        results_file = f'{args.dataset}_{time_prefix}_fs{args.few_shot}_s{args.seed}.json'
        json.dump(merged_outputs, open(results_file, 'w'), ensure_ascii=False)

        if ds_collections[args.dataset]['metric'] == 'vqa_score':
            vqa = VQA(ds_collections[args.dataset]['annotation'],
                      ds_collections[args.dataset]['question'])
            results = vqa.loadRes(
                resFile=results_file,
                quesFile=ds_collections[args.dataset]['question'])
            vqa_scorer = VQAEval(vqa, results, n=2)
            vqa_scorer.evaluate()

            print(vqa_scorer.accuracy)
    torch.distributed.barrier()
