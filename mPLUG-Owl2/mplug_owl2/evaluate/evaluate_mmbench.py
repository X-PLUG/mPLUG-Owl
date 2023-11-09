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
import pandas as pd

from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mplug_owl2.conversation import conv_templates, SeparatorStyle
from mplug_owl2.model.builder import load_pretrained_model
from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


ds_collections = {
    'mmbench_dev_20230712': {
        'raw_file': 'mmbench_dev_20230712.tsv',
        'annotation': 'mmbench_dev_20230712.jsonl',
        'max_new_tokens': 10,
    },
    'mmbench_test_20230712': {
        'raw_file': 'mmbench_test_20230712.tsv',
        'annotation': 'mmbench_test_20230712.jsonl',
        'max_new_tokens': 10,
    },
}

multiple_choices = ['A', 'B', 'C', 'D', 'E']

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
            "l2-category": row_df['l2-category']
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

def collate_fn(batches, tokenizer):

    questions = [_['question'] for _ in batches]
    indices = [_['index'] for _ in batches]

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
    return image_tensor, input_ids, attention_mask, indices


class VQADataset(torch.utils.data.Dataset):

    def __init__(self, test, prompt, image_processor):
        
        self.prompt = prompt
        self.image_processor = image_processor

        self.data = open(test).readlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = json.loads(self.data[idx].strip())
        index = data['index']
        image = data['image']
        hint = data['hint'] if data['hint'] else 'N/A'
        question = data['question']

        choices = data['choices']
        choice_list = []
        for i, c in enumerate(choices):
            choice_list.append('{}. {}'.format(multiple_choices[i], c))
        choice_txt = '\n'.join(choice_list)

        image = Image.open(image).convert('RGB')
        max_edge = max(image.size)
        image = image.resize((max_edge, max_edge)) # Resize here for best performance
        image_tensor = process_images([image], self.image_processor)

        return {
            'index': index,
            'image_tensor': image_tensor,
            'question': self.prompt.format(hint, question, choice_txt),
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
    parser.add_argument('--dataset', type=str, default='mmbench_dev_20230712')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))
    os.environ['CUDA_VISIBLE_DEVICES'] = os.getenv('LOCAL_RANK', "0")

    model_path = args.checkpoint
    model_name = get_model_name_from_path(model_path)
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit=False, load_4bit=False, device_map="cuda", device="cuda")
    tokenizer.padding_side = 'left'
    tokenizer.pad_token_id = tokenizer.eos_token_id

    prompt = "USER: <|image|>{}\n{}\n{}\nAnswer with the option’s letter from the given choices directly. ASSISTANT:"

    random.seed(args.seed)
    dataset = VQADataset(
        test=ds_collections[args.dataset]['annotation'],
        prompt=prompt,
        image_processor=image_processor,
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
    for _, (image_tensor, input_ids, attention_mask, indices) in tqdm(enumerate(dataloader)):
        pred = model.generate(
            input_ids=input_ids.cuda(),
            attention_mask=attention_mask.cuda(),
            images=image_tensor.to(dtype=model.dtype).cuda(),
            do_sample=False,
            num_beams=1,
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

        for index, answer in zip(indices, answers):
            outputs.append({
                'index': index,
                'prediction': answer,
            })

    torch.distributed.barrier()

    world_size = torch.distributed.get_world_size()
    merged_outputs = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))

    merged_outputs = [json.loads(_) for _ in merged_outputs]
    merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

    if torch.distributed.get_rank() == 0:
        print(f"Evaluating {args.dataset} ...")
        time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
        results_file = f'{args.dataset}_{time_prefix}_s{args.seed}.json'
        json.dump(merged_outputs, open(results_file, 'w'), ensure_ascii=False)

        mapped_result = mapping_to_annotation(merged_outputs, pd.read_csv(ds_collections[args.dataset]['raw_file'], sep='\t'))

        submission_res = generate_submission_file(merged_outputs, pd.read_csv(ds_collections[args.dataset]['raw_file'], sep='\t'))
        res_df = pd.DataFrame(submission_res)
        metrics_file = f'{args.dataset}_{time_prefix}_s{args.seed}_submission.xlsx'
        res_df.to_excel(metrics_file, index=False)

    torch.distributed.barrier()
