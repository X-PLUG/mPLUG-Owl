import json
import logging
import os
import random
import re
import time
import traceback
import warnings
from io import BytesIO

import h5py
import numpy as np
import torch
from icecream import ic
from PIL import Image, ImageFile
from torch.utils.data import Dataset, Subset

from utils import get_args

from .processors import build_processors

ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def load_jsonl(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]


class MultiModalDataset(Dataset):
    """MultiModal dataset"""

    def __init__(self, input_files, tokenizer, processors,
                 max_length=2048,
                 media_tokens=['<image>']):
        args = get_args()
        self.dataset = []
        if isinstance(input_files, str):
            input_files = [input_files]
        for input_file in input_files:
            self.dataset += load_jsonl(input_file)

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.processors = processors
        self.media_tokens = {k: -int(i+1) for i, k in enumerate(media_tokens)}
        self.media_lengths = {'<image>': 1+64}
        print("num_media_token: ", self.media_lengths)
        self.bucket = {}
        print(len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def _load_img(self, images):
        if isinstance(images, str):
            images = [images]
        image_pils = []
        for image_url in images:
            image = Image.open(image_url).convert('RGB')
            image_pils.append(image)
        return image_pils

    def process_data(self, data, processor=None):
        # Process Image if exists
        if 'image' in data and len(data['image']) > 0:
            if 'image_data' in data:
                images = data['image_data']
            else:
                image_urls = data['image']
                images = self._load_img(image_urls)
            if processor:
                images = [processor(image=image, text=None)[0]
                          for image in images]
                images = torch.stack(images, dim=0)
        else:
            images = None

        # Process Text
        text = {
            "prompt": data.get('prompt', ""),
            "text": data["text"]
        }
        if processor:
            text = processor(image=None, text=text)[1]
        return images, text


    def __getitem__(self, index):
        data = self.dataset[index]
        task_type = data.get('task_type', 'dummy_default').split(
            '_')[-1]  # Get processor type
        while True:
            try:
                # use for processing image-text pairs
                image, text = self.process_data(
                    data, self.processors[task_type])

                text_input = self._extract_text_token_from_conversation(
                    text, self.max_length, index)
        
            except Exception as e:
                traceback.print_exc()
                # print(e)
                #logging.info("Get image:{} from oss failed, retry.".format(ann))
                time.sleep(0.1)
                index = 0 if index == (len(self) - 1) else index + 1
                data = self.dataset[index]
                task_type = data.get(
                    'task_type', 'dummy_default').split('_')[-1]
                continue
            break

        batch_data = {
            "image": image,
            "text": text_input
        }

        return batch_data

    def _extract_text_token_from_conversation(self, data, max_length, index):
        # output enc_chunk
        enc_chunk = []

        if self.tokenizer.bos_token_id > 0:
            prompt_chunk = [self.tokenizer.bos_token_id]
        else:
            prompt_chunk = []

        conversation = data["completion"]
        # For Text only data
        if all([media_token not in conversation for media_token in self.media_tokens.keys()]):
            pattern = '|'.join(map(re.escape, ['AI: ', '\nHuman: ']))
            chunk_strs = re.split(f'({pattern})', conversation)
            prompt_length = -1
            stop_flag = False
            for idx, chunk_str in enumerate(chunk_strs):
                if idx == 0:
                    enc_chunk = prompt_chunk + \
                        self.tokenizer(chunk_str, add_special_tokens=False)[
                            'input_ids']
                    enc_length = len(enc_chunk)
                    label_chunk = [0] * enc_length
                else:
                    if chunk_strs[idx-1] == 'AI: ':
                        curr_chunk = self.tokenizer(
                            chunk_str, add_special_tokens=False)['input_ids']
                        if enc_length + len(curr_chunk) >= max_length:
                            curr_chunk = curr_chunk[:max_length-enc_length]
                            stop_flag = True
                        curr_chunk += [self.tokenizer.eos_token_id]
                        enc_length += len(curr_chunk)
                        enc_chunk += curr_chunk
                        label_chunk += [1] * len(curr_chunk)
                    else:
                        curr_chunk = self.tokenizer(
                            chunk_str, add_special_tokens=False)['input_ids']
                        if enc_length + len(curr_chunk) >= max_length + 1:
                            curr_chunk = curr_chunk[:max_length+1-enc_length]
                            stop_flag = True
                        enc_length += len(curr_chunk)
                        enc_chunk += curr_chunk
                        label_chunk += [0] * len(curr_chunk)
                    if stop_flag:
                        break

        # For Image-Text Data
        else:
            enc_length = 0
            prompt_length = -2
            pattern = '|'.join(
                map(re.escape, list(self.media_tokens.keys()) + ['AI: ', '\nHuman: ']))
            chunk_strs = re.split(f'({pattern})', conversation)
            chunk_strs = [x for x in chunk_strs if len(x) > 0]
            for idx, chunk_str in enumerate(chunk_strs):
                if enc_length >= max_length + 1:
                    break

                if idx == 0:
                    enc_chunk = prompt_chunk + \
                        self.tokenizer(chunk_str, add_special_tokens=False)[
                            'input_ids']
                    enc_length = len(enc_chunk)
                    label_chunk = [0] * enc_length
                else:
                    if chunk_str in self.media_tokens:
                        # [CLS] + 256 + [EOS]
                        if enc_length + self.media_lengths[chunk_str] > max_length + 1:
                            break
                        else:
                            enc_chunk += [self.media_tokens[chunk_str]
                                          ] * self.media_lengths[chunk_str]
                            enc_length += self.media_lengths[chunk_str]
                            label_chunk += [0] * self.media_lengths[chunk_str]
                    else:

                        if chunk_strs[idx-1] == 'AI: ':
                            curr_chunk = self.tokenizer(
                                chunk_str, add_special_tokens=False)['input_ids']
                            if enc_length + len(curr_chunk) >= max_length:
                                curr_chunk = curr_chunk[:max_length-enc_length]
                            curr_chunk += [self.tokenizer.eos_token_id]
                            enc_length += len(curr_chunk)
                            enc_chunk += curr_chunk
                            label_chunk += [1] * len(curr_chunk)
                        else:
                            curr_chunk = self.tokenizer(
                                chunk_str, add_special_tokens=False)['input_ids']
                            if enc_length + len(curr_chunk) >= max_length + 1:
                                curr_chunk = curr_chunk[:max_length +
                                                        1-enc_length]
                            enc_length += len(curr_chunk)
                            enc_chunk += curr_chunk
                            label_chunk += [0] * len(curr_chunk)

        if enc_length < max_length + 1:
            padding_chunk = [self.tokenizer.pad_token_id] * \
                (max_length + 1 - enc_length)
            padding_length = len(padding_chunk)
            label_chunk += [0] * (max_length + 1 - enc_length)
            enc_chunk = enc_chunk + padding_chunk
        else:
            padding_length = 0

        assert enc_length + padding_length == max_length + \
            1, (index, prompt_length, enc_length,
                padding_length, max_length + 1)
        assert len(label_chunk) == max_length + \
            1, (len(label_chunk), max_length + 1)
        non_padding_mask = [1 if i < enc_length -
                            1 else 0 for i in range(max_length)]

        enc_chunk = torch.tensor(enc_chunk).long()
        non_padding_mask = torch.tensor(non_padding_mask).long()
        prompt_mask = torch.tensor(label_chunk)[1:].long()
        prompt_length = torch.tensor([prompt_length]).long()

        # Create loss mask
        if all([media_token not in conversation for media_token in self.media_tokens.keys()]):
            non_media_mask = torch.ones_like(non_padding_mask).long()
        else:
            tmp_enc_chunk = enc_chunk.clone()
            tmp_enc_chunk[tmp_enc_chunk >= 0] = 1
            tmp_enc_chunk[tmp_enc_chunk < 0] = 0
            non_media_mask = torch.tensor(tmp_enc_chunk).long()
            non_media_mask = non_media_mask[1:].long()
        return {'input_ids': enc_chunk, "prompt_length": prompt_length, 'seq_length': enc_length,
                "non_padding_mask": non_padding_mask, 'non_media_mask': non_media_mask, 'prompt_mask': prompt_mask}