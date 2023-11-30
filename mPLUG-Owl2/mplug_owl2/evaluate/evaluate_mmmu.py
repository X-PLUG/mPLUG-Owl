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
import re

from datasets import load_dataset

from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mplug_owl2.conversation import conv_templates, SeparatorStyle
from mplug_owl2.model.builder import load_pretrained_model
from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

DOMAIN_CAT2SUB_CAT = {
  'Art and Design': ['Art', 'Art_Theory', 'Design', 'Music'],
  'Business': ['Accounting', 'Economics', 'Finance', 'Manage','Marketing'],
  'Science': ['Biology', 'Chemistry', 'Geography', 'Math', 'Physics',],
  'Health and Medicine': ['Basic_Medical_Science', 'Clinical_Medicine', 'Diagnostics_and_Laboratory_Medicine', 'Pharmacy', 'Public_Health'],
  'Humanities and Social Science': ['History', 'Literature', 'Sociology', 'Psychology'],
  'Tech and Engineering': ['Agriculture', 'Architecture_and_Engineering', 'Computer_Science', 'Electronics', 'Energy_and_Power', 'Materials', 'Mechanical_Engineering'],
}


CAT_SHORT2LONG = {
    'acc': 'Accounting',
    'agri': 'Agriculture',
    'arch': 'Architecture_and_Engineering',
    'art': 'Art',
    'art_theory': 'Art_Theory',
    'bas_med': 'Basic_Medical_Science',
    'bio': 'Biology',
    'chem': 'Chemistry',
    'cli_med': 'Clinical_Medicine',
    'cs': 'Computer_Science',
    'design': 'Design',
    'diag_med': 'Diagnostics_and_Laboratory_Medicine',
    'econ': 'Economics',
    'elec': 'Electronics',
    'ep': 'Energy_and_Power',
    'fin': 'Finance',
    'geo': 'Geography',
    'his': 'History',
    'liter': 'Literature',
    'manage': 'Manage',
    'mark': 'Marketing',
    'mate': 'Materials',
    'math': 'Math',
    'mech': 'Mechanical_Engineering',
    'music': 'Music',
    'phar': 'Pharmacy',
    'phys': 'Physics',
    'psy': 'Psychology',
    'pub_health': 'Public_Health',
    'socio': 'Sociology'
}


multiple_choices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']


# ----------- Process Multi-choice -------------
def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    """
    for char in [',', '.', '!', '?', ';', ':', "'"]:
        response = response.strip(char)
    response = " " + response + " " # add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f'({choice})' in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices: # e.g., A B C D
            if f' {choice} ' in response:
                candidates.append(choice)

    # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False # it's content ans.

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack: 
                for can in candidates:
                    index = response.rfind(f'({can})')
                    start_indexes.append(index) # -1 will be ignored anyway
                # start_indexes = [generated_response.index(f'({can})') for can in candidates]
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        # get the last one
        pred_index = candidates[np.argmax(start_indexes)]
    else: # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index

# ----------- Process Open -------------
def check_is_number(string):
    """
    Check if the given string a number.
    """
    try:
        float(string.replace(',', ''))
        return True
    except ValueError:
        # check if there's comma inside
        return False

def normalize_str(string):
    """
    Normalize the str to lower case and make them float numbers if possible.
    """
    # check if characters in the string

    # if number, numerize it.
    string = string.strip()

    is_number = check_is_number(string)

    if is_number:
        string = string.replace(',', '')
        string = float(string)
        # leave 2 decimal
        string = round(string, 2)
        return [string]
    else: # it's likely to be a string
        # lower it 
        string = string.lower()
        if len(string) == 1:
            return [" " + string, string + " "] # avoid trivial matches
        return [string]

def extract_numbers(string):
    """
    Exact all forms of numbers from a string with regex.
    """
    # Pattern for numbers with commas
    pattern_commas = r'-?\b\d{1,3}(?:,\d{3})+\b'
    # Pattern for scientific notation
    pattern_scientific = r'-?\d+(?:\.\d+)?[eE][+-]?\d+'
    # Pattern for simple numbers without commas
    pattern_simple = r'-?(?:\d+\.\d+|\.\d+|\d+\b)(?![eE][+-]?\d+)(?![,\d])'

    # Extract numbers with commas
    numbers_with_commas = re.findall(pattern_commas, string)
    # Extract numbers in scientific notation
    numbers_scientific = re.findall(pattern_scientific, string)
    # Extract simple numbers without commas
    numbers_simple = re.findall(pattern_simple, string)

    # Combine all extracted numbers
    all_numbers = numbers_with_commas + numbers_scientific + numbers_simple
    return all_numbers

def parse_open_response(response):
    """
    Parse the prediction from the generated response.
    Return a list of predicted strings or numbers.
    """
    # content = content.strip("\n").strip(".").strip(" ")
    def get_key_subresponses(response):
        key_responses = []
        response = response.strip().strip(".").lower()
        sub_responses = re.split(r'\.\s(?=[A-Z])|\n', response)
        indicators_of_keys = ['could be ', 'so ', 'is ',
                            'thus ', 'therefore ', 'final ', 'answer ', 'result ']
        key_responses = []
        for index, resp in enumerate(sub_responses):
            # if last one, accept it's an equation (the entire response can be just one sentence with equation)
            if index == len(sub_responses) - 1:
                indicators_of_keys.extend(['='])
            shortest_key_response = None # the shortest response that may contain the answer (tail part of the response)
            for indicator in indicators_of_keys:
                if indicator in resp:
                    if not shortest_key_response:
                        shortest_key_response = resp.split(indicator)[-1].strip()
                    else:
                        if len(resp.split(indicator)[-1].strip()) < len(shortest_key_response):
                            shortest_key_response = resp.split(indicator)[-1].strip()
                    # key_responses.append(resp.split(indicator)[1].strip())
            
            if shortest_key_response:
                # and it's not trivial
                if shortest_key_response.strip() not in [":", ",", ".", "!", "?", ";", ":", "'"]:
                    key_responses.append(shortest_key_response)
        if len(key_responses) == 0: # did not found any
            return [response]
        return key_responses
    # pdb.set_trace()
    key_responses = get_key_subresponses(response)

    pred_list = key_responses.copy() # keep the original string response
    for resp in key_responses:
        pred_list.extend(extract_numbers(resp))

    tmp_pred_list = []
    for i in range(len(pred_list)):
        tmp_pred_list.extend(normalize_str(pred_list[i]))
    pred_list = tmp_pred_list

    # remove duplicates
    pred_list = list(set(pred_list))

    return pred_list

# ----------- Evaluation -------------

def eval_multi_choice(gold_i, pred_i):
    """
    Evaluate a multiple choice instance.
    """
    correct = False
    # only they are exactly the same, we consider it as correct
    if isinstance(gold_i, list):
        for answer in gold_i:
            if answer == pred_i:
                correct = True
                break
    else: # gold_i is a string
        if gold_i == pred_i:
            correct = True
    return correct

def eval_open(gold_i, pred_i):
    """
    Evaluate an open question instance
    """
    correct = False
    if isinstance(gold_i, list):
        # use float to avoid trivial matches
        norm_answers = []
        for answer in gold_i:
            norm_answers.extend(normalize_str(answer))
    else:
        norm_answers = normalize_str(gold_i)
    for pred in pred_i: # pred is already normalized in parse response phase
        if isinstance(pred, str): # if it's a string, then find if ans in the pred_i
            for norm_ans in norm_answers:
                # only see if the string answer in the string pred
                if isinstance(norm_ans, str) and norm_ans in pred:
                    if not correct:
                        correct = True
                    break
        else: # it's a float number
            if pred in norm_answers:
                if not correct:
                    correct = True
                break
    return correct


def evaluate(samples):
    """
    Batch evaluation for multiple choice and open questions.
    """
    pred_correct = 0
    judge_dict = dict()
    for sample in samples:
        gold_i = sample['ground_truth']
        pred_i = sample['prediction']
        if sample['question_type'] == 'multiple-choice':
            correct = eval_multi_choice(gold_i, pred_i)
        else: # open question
            correct = eval_open(gold_i, pred_i)

        if correct:
            judge_dict[sample['index']] = 'Correct'
            pred_correct += 1
        else:
            judge_dict[sample['index']] = 'Wrong'

    if len(samples) == 0:
        return {'acc': 0}
    return judge_dict, {'acc': pred_correct / len(samples)}


def collate_fn(batches, tokenizer):
    origin_questions = [_['origin_question'] for _ in batches]
    question_types = [_['question_type'] for _ in batches]
    answers = [_['answer'] for _ in batches]
    questions = [_['question'] for _ in batches]
    ids = [_['id'] for _ in batches]
    subfields = [_['subfield'] for _ in batches]
    question_splits = [_['question_split'] for _ in batches]

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
    return image_tensor, input_ids, attention_mask, answers, ids, origin_questions, question_types, subfields, question_splits


class VQADataset(torch.utils.data.Dataset):

    def __init__(self, split, image_processor, eval_split='dev'):
        
        self.image_processor = image_processor
        self.data = load_dataset("MMMU/MMMU", split)[eval_split]
        self.question_split = split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data.__getitem__(idx)
        name_id = data['id']
        question = data['question']
        subfield = data['subfield']
        question_type = data['question_type']
        answer = data['answer']

        if question_type == 'multiple-choice':
            choices = eval(data['options'])
            choice_list = []
            for i, c in enumerate(choices):
                choice_list.append('{}. {}'.format(multiple_choices[i], c))
            choice_txt = '\n'.join(choice_list)
            prompt = f"USER: {question}\n{choice_txt}\nAnswer with the option’s letter from the given choices directly. ASSISTANT:"
        else:
            prompt = f"USER: {question}\nAnswer the question using a single word or phrase. ASSISTANT:"
        
        image_nums = re.findall(r'<image (\d+)>', prompt)
        images = []
        for image_num in image_nums:
            image = data[f'image_'+image_num].convert('RGB')
            max_edge = max(image.size)
            image = image.resize((max_edge, max_edge))
            images.append(image)

        for i in range(1, 6):
            prompt = prompt.replace(f'<image {i}>', '<|image|>')

        image_tensor = process_images(images, self.image_processor)

        return {
            'id': name_id,
            'subfield': subfield,
            'origin_question': question,
            'answer': answer,
            'question_type': question_type,
            'question_split': self.question_split,
            'question': prompt,
            'image_tensor': image_tensor,
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
    parser.add_argument('--dataset', type=str, default='dev')
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


    random.seed(args.seed)
    datasets = []
    for split in CAT_SHORT2LONG.values():
        datasets.append(VQADataset(
            split, image_processor, args.dataset
        ))
    dataset = torch.utils.data.ConcatDataset(datasets)

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
    for _, (image_tensor, input_ids, attention_mask, labels, ids, origin_questions, question_types, subfields, question_splits) in enumerate(tqdm(dataloader)):
        pred = model.generate(
            input_ids=input_ids.cuda(),
            attention_mask=attention_mask.cuda(),
            images=image_tensor.to(dtype=model.dtype).cuda(),
            do_sample=False,
            num_beams=1,
            max_new_tokens=20,
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

        for d_ in zip(ids, answers, labels, origin_questions, question_types, subfields, question_splits):
            index, answer, label, origin_question, question_type, subfield, question_split = d_
            outputs.append({
                'index': index,
                'prediction': answer,
                'ground_truth': label,
                'split': question_split,
                'subfield': subfield,
                'question_type': question_type,
                'origin_question': origin_question,
            })
    print('finish.')
    torch.distributed.barrier()

    world_size = torch.distributed.get_world_size()
    merged_outputs = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))

    merged_outputs = [json.loads(_) for _ in merged_outputs]
    merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

    if torch.distributed.get_rank() == 0:
        print(f"Evaluating {args.dataset} ...")
        time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
        results_file = f'{args.dataset}_{time_prefix}_s{args.seed}_results.json'
        metrics_file = f'{args.dataset}_{time_prefix}_s{args.seed}_metrics.json'
        predict_file = f'{args.dataset}_{time_prefix}_s{args.seed}_evaluation.json'
        submit_file = f'{args.dataset}_{time_prefix}_s{args.seed}_prediction_groupby_category.json'
        json.dump(merged_outputs, open(results_file, 'w'), ensure_ascii=False)

        groupby_category = {}
        for output in merged_outputs:
            if output['split'] not in groupby_category:
                groupby_category[output['split']] = []
            output_tmp = output.copy()
            if output_tmp['question_type'] == 'multiple-choice':
                pass
            else:
                output_tmp['prediction'] = parse_open_response(output_tmp['prediction'])

            groupby_category[output['split']].append(output_tmp)

        json.dump(groupby_category, open(submit_file, 'w'), ensure_ascii=False)

        if args.dataset in ['dev', 'validation']:
            metrics = {}
            metrics['Overall'] = 0
            evaluation_results = {}
            for category, outputs_per_category in groupby_category.items():
                evaluation_result, metric = evaluate(outputs_per_category)
                metrics[category] = metric['acc'] * 100
                evaluation_results[category] = evaluation_result
                metrics['Overall'] += metric['acc'] * len(outputs_per_category)
            metrics['Overall'] /= len(merged_outputs)
            metrics['Overall'] *= 100

            json.dump(metrics, open(metrics_file, 'w'), ensure_ascii=False)
            json.dump(evaluation_results, open(predict_file, 'w'), ensure_ascii=False)

    torch.distributed.barrier()
