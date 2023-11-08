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

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

ds_collections = {
    'mme': {
        "test": "MME_Benchmark_release_version/eval_tool/Your_Results",
        "base_dir": 'MME_Benchmark_release_version',
        'max_new_tokens': 10,
    },
}


eval_type_dict = {
    "Perception": ["existence", "count", "position", "color", "posters", "celebrity", "scene", "landmark", "artwork", "OCR"],
    "Cognition": ["commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning"]
}


class calculate_metrics:
    def divide_chunks(self, l, n=2):
        # looping till length l
        for i in range(0, len(l), n): 
            yield l[i:i + n]
        
        return 

    def parse_pred_ans(self, pred_ans):
        pred_label = None
        if pred_ans in ["yes", "no"]:
            pred_label = pred_ans
        else:
            prefix_pred_ans = pred_ans[:4]

            if "yes" in prefix_pred_ans:
                pred_label = "yes"
            elif "no" in prefix_pred_ans:
                pred_label = "no"
            else:
                pred_label = "other"

        return pred_label


    def compute_metric(self, gts, preds):
        assert len(gts) == len(preds)

        label_map = {
            "yes": 1,
            "no": 0,
            "other": -1,
        }
        
        gts = [label_map[x] for x in gts]
        preds = [label_map[x] for x in preds]

        acc = accuracy_score(gts, preds) 

        clean_gts = []
        clean_preds = []
        other_num = 0 
        for gt, pred in zip(gts, preds):
            if pred == -1:
                other_num += 1
                continue
            clean_gts.append(gt)
            clean_preds.append(pred)
        

        conf_mat = confusion_matrix(clean_gts, clean_preds, labels=[1,0])
        precision = precision_score(clean_gts, clean_preds, average='binary')
        recall = recall_score(clean_gts, clean_preds, average='binary')
        tp, fn = conf_mat[0]
        fp, tn = conf_mat[1]

        metric_dict = dict()
        metric_dict = {
            "TP": tp,
            "FN": fn,
            "TN": tn,
            "FP": fp,
            "precision": precision,
            "recall": recall,
            "other_num": other_num,
            "acc": acc,
        }

        return metric_dict


    def process_result(self, outputs):

        model_score_dict = dict()
        for eval_type, task_name_list in eval_type_dict.items():
            print("===========", eval_type, "===========")           
            scores = 0
            task_score_dict = dict()

            for task_name in task_name_list:
                chunk_lines = outputs[task_name]                
                img_num = len(chunk_lines)
                task_other_ans_num = 0
                task_score = 0
                acc_plus_correct_num = 0
                gts = []
                preds = []

                for k, img_items in chunk_lines.items():
                    assert len(img_items) == 2
                    img_correct_num = 0

                    for img_item in img_items:
                        img_name, question, gt_ans, pred_ans = img_item['image_name'], img_item['question'], img_item['annotation'], img_item['answer']
                        gt_ans = gt_ans.lower()
                        pred_ans = pred_ans.lower()

                        assert gt_ans in ["yes", "no"] # gt can only be yes or no.

                        pred_ans = self.parse_pred_ans(pred_ans)
                        assert pred_ans in ["yes", "no", "other"]

                        gts.append(gt_ans)
                        preds.append(pred_ans)
                        
                        if gt_ans == pred_ans:
                            img_correct_num += 1
                        
                        if pred_ans not in ["yes", "no"]:
                            task_other_ans_num += 1

                    if img_correct_num == 2:
                        acc_plus_correct_num += 1

                # cal TP precision acc, etc.
                metric_dict = self.compute_metric(gts, preds)
                acc_plus = acc_plus_correct_num / img_num
                metric_dict["acc_plus"] = acc_plus
                
                
                for k, v in metric_dict.items():
                    if k in ["acc", "acc_plus"]:
                        task_score += v*100
                
                task_score_dict[task_name] = task_score
                
                scores += task_score
                
            print("total score:", scores, "\n")
            for task_name, score in task_score_dict.items():
                print("\t", task_name, " score:", score)
                model_score_dict[eval_type+"-"+task_name] = score
            print("\n")
        
        return model_score_dict

def collate_fn(batches, tokenizer):

    questions = [_['question'] for _ in batches]
    questions_origin = [_['question_origin'] for _ in batches]
    annotations = [_['annotation'] for _ in batches]
    image_names = [_['image_name'] for _ in batches]
    categories = [_['category'] for _ in batches]

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
    return questions_origin, image_tensor, input_ids, attention_mask, annotations, categories, image_names


class VQADataset(torch.utils.data.Dataset):

    def __init__(self, test, base_dir, prompt, image_processor):
        
        self.prompt = prompt
        self.image_processor = image_processor

        self.categories = [
            'OCR', 'artwork', 'celebrity',
            'code_reasoning', 'color',
            'commonsense_reasoning','count',
            'existence', 'landmark',
            'numerical_calculation', 'position',
            'posters', 'scene', 'text_translation'
        ]
        self.data = []
        self.base_dir = base_dir
        for category in self.categories:
            anno_path = os.path.join(test, '{}.txt'.format(category))
            with open(anno_path, 'r') as f:
                self.data += [(x, category) for x in f.read().splitlines()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        d, category = data
        image_name, question, annotation = d.split('\t')

        category_path = os.path.join(self.base_dir, category)
        if category in ['landmark', 'artwork', 'celebrity', 'posters', 'scene']:
            image_path = os.path.join(category_path, 'images')
            text_path = os.path.join(category_path, 'questions_answers_YN')
        else:
            image_path = category_path
            text_path = category_path
        image = os.path.join(image_path, image_name)

        image = Image.open(image).convert('RGB')
        max_edge = max(image.size)
        image = image.resize((max_edge, max_edge)) # Resize here for best performance
        image_tensor = process_images([image], self.image_processor)

        return {
            'image_tensor': image_tensor,
            'question': self.prompt.format(question).replace('Please answer yes or no.', ''),
            "category": category,
            "image_name": image_name,
            'question_origin': question,
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
    parser.add_argument('--dataset', type=str, default='mme')
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

    prompt = 'USER: <|image|>{}\nAnswer the question using a single word or phrase. ASSISTANT:'

    random.seed(args.seed)
    dataset = VQADataset(
        test=ds_collections[args.dataset]['test'],
        base_dir=ds_collections[args.dataset]['base_dir'],
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
    for _, (questions, image_tensor, input_ids, attention_mask,
            annotations, categories, image_names) in tqdm(enumerate(dataloader)):
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

        for question, answer, annotation, category, image_name in zip(questions, answers,
                                                   annotations, categories, image_names):
            outputs.append({
                'image_name': image_name,
                'category': category,
                'question': question,
                'answer': answer,
                'annotation': annotation,
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

        
        groups = {}
        for output in merged_outputs:
            if not output['category'] in groups:
                groups[output['category']] = {}
            if not output['image_name'] in groups[output['category']]:
                groups[output['category']][output['image_name']] = []

            groups[output['category']][output['image_name']].append({
                "image_name": output['image_name'],
                "question": output['question'],
                "answer": output['answer'],
                "annotation": output['annotation'],
            })

        cal = calculate_metrics()
        model_score_dict = cal.process_result(groups)
        metrics = {
            "Perception": sum([v for k, v in model_score_dict.items() if 'Perception' in k]),
            "Cognition": sum([v for k, v in model_score_dict.items() if 'Cognition' in k]),
            **model_score_dict
        }
        print(metrics)

    torch.distributed.barrier()
