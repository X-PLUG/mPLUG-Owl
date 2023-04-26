# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sample Generate GPT"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import socket
from megatron import get_args
from megatron import get_tokenizer
from megatron import print_rank_0
from megatron import mpu
from megatron.checkpointing import load_checkpoint
from megatron.initialize import initialize_megatron
from megatron.training import get_model
from megatron.model import XGPT3Model
from megatron.text_generation_xgpt import generate_and_post_process
from megatron.text_generation_xgpt import beam_search_and_post_process
import torch

import datetime
import torch
import json
import threading
# from flask import Flask, request, jsonify, current_app
# from flask_restful import Resource, Api

import utils
from pathlib import Path
import ruamel.yaml as yaml
from icecream import ic

from torchvision import transforms
from PIL import Image
import json
import requests
from io import BytesIO
import base64
import re


GENERATE_NUM = 0
BEAM_NUM = 1
lock = threading.Lock()

import json
from waitress import serve
from flask import Flask, request, make_response
# from flask_cors import cross_origin

TOKENIZER = None
CHAT_MODEL = None
NO_DEBUG = False
INSTANCE_CODE = None

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
# @cross_origin()
def process_handle_func():
    data = request.get_data() #.decode('utf-8')
    body = json.loads(data)
    res = process(body)
    response = make_response(res)
    response.status_code = 200
    return response

@app.route('/health_check', methods=['GET', 'POST'])
def health_check():
    response = make_response({'success': True})
    response.status_code = 200
    return response

def process(input):
    prompt = input['text_input']
    images = input.get('images', None)
    generate_config = input.get('generate_config', json.dumps(
        {"top_k": 5, 
        "beam_width": 3, 
        "stop_token": 1,
        "tokens_to_generate": 256
        }, ensure_ascii=False))

    params = json.loads(generate_config)
    params["prompts"] = [prompt]
    if images:
        params["images"] = images

    r = Request(params)
    global CHAT_MODEL
    response = CHAT_MODEL.inference(r)
    global TOKENIZER
    eod_token = input.get("stop_token", TOKENIZER.eod_token)

    # post process
    if isinstance(response, str):
        response_tmp = [response]
    else:
        response_tmp = response
    responses = []
    for response in response_tmp:
        if eod_token in response:
            index = response.index(eod_token)
            response = response[:index]
        if TOKENIZER.eod_token in response:
            index = response.index(TOKENIZER.eod_token)
            response = response[:index]
        responses.append(response)
    if len(responses) == 1:
        responses = responses[0]
    return responses


class Request(object):

    def __init__(self, params):
        self.params = params

    def get_json(self):
        return self.params


def base64decode(s: str):
    """
    Decode base64 `str` to original `bytes`.
    If the input is not a valid base64 string, return None.

    Args:
        s(str): A base64 `str` that can be used in text file.

    Returns:
        Optional[bytes]: The original decoded data with type `bytes`.
            If the input is not a valid base64 string, return None.
    """
    # return base64.b64decode(s)
    _base64_regex = re.compile(r'^(?:[A-Za-z\d+/]{4})*(?:[A-Za-z\d+/]{3}=|[A-Za-z\d+/]{2}==)?$')
    s = s.translate(base64._urlsafe_decode_translation)
    if not _base64_regex.fullmatch(s):
        return None
    try:
        return base64.urlsafe_b64decode(s)
    except base64.binascii.Error:
        return None




class mPLUGOwlGenerate():
    def __init__(self, model, tokenizer, img_processor):
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = img_processor

    @staticmethod
    def send_do_generate():
        choice = torch.cuda.LongTensor([GENERATE_NUM])
        torch.distributed.broadcast(choice, 0)
     
    @staticmethod
    def send_do_beam_search():
        choice = torch.cuda.LongTensor([BEAM_NUM])
        torch.distributed.broadcast(choice, 0)
    
    def inference(self, request):
        args = get_args()
       
        if not "prompts" in request.get_json():
            return "prompts argument required", 400
        
        if "max_len" in request.get_json():
            return "max_len is no longer used.  Replace with tokens_to_generate", 400
        
        if "sentences" in request.get_json():
            return "sentences is no longer used.  Replace with prompts", 400

        prompts = request.get_json()["prompts"]
        
        # prompts = [prompt.replace("<enter>", "\n") for prompt in prompts]
        # print(prompts)
        if len(prompts) > 128:
            return "Maximum number of prompts is 128", 400

        tokens_to_generate = 64  # Choosing hopefully sane default.  Full sequence is slow
        if "tokens_to_generate" in request.get_json():
            tokens_to_generate = request.get_json()["tokens_to_generate"]
            if not isinstance(tokens_to_generate, int):
                return "tokens_to_generate must be an integer greater than 0"
            if tokens_to_generate < 0:
                return "tokens_to_generate must be an integer greater than or equal to 0"

        images = None
        if "images" in request.get_json():
            images = request.get_json()["images"]
            if not isinstance(images, list) and not isinstance(images, str):
                return "Images has to be list or list of strings"
            elif isinstance(images, list) and any([not isinstance(image, str) for image in images]):
                return "The Image url in the list should be string"

        logprobs = False
        if "logprobs" in request.get_json():
            logprobs = request.get_json()["logprobs"]
            if not isinstance(logprobs, bool):
                return "logprobs must be a boolean value"
        
        if tokens_to_generate == 0 and not logprobs:
            return "tokens_to_generate=0 implies logprobs should be True"
        
        temperature = 1.0
        if "temperature" in request.get_json():
            temperature = request.get_json()["temperature"]
            if not (type(temperature) == int or type(temperature) == float):
                return "temperature must be a positive number less than or equal to 100.0"
            if not (0.0 < temperature <= 100.0):
                return "temperature must be a positive number less than or equal to 100.0"
        
        top_k = 0.0
        if "top_k" in request.get_json():
            top_k = request.get_json()["top_k"]
            if not (type(top_k) == int):
                return "top_k must be an integer equal to or greater than 0 and less than or equal to 1000"
            if not (0 <= top_k <= 1000):
                return "top_k must be equal to or greater than 0 and less than or equal to 1000"
        
        top_p = 0.0
        if "top_p" in request.get_json():
            top_p = request.get_json()["top_p"]
            if not (type(top_p) == float):
                return "top_p must be a positive float less than or equal to 1.0"
            if top_p > 0.0 and top_k > 0.0:
                return "cannot set both top-k and top-p samplings."
            if not (0 <= top_p <= 1.0):
                return "top_p must be less than or equal to 1.0"
        
        add_BOS = False
        if "add_BOS" in request.get_json():
            add_BOS = request.get_json()["add_BOS"]
            if not isinstance(add_BOS, bool):
                return "add_BOS must be a boolean value"
        
        if any([len(prompt) == 0 for prompt in prompts]) and not add_BOS:
            return "Empty prompts require add_BOS=true"

        stop_on_double_eol = False
        if "stop_on_double_eol" in request.get_json():
            stop_on_double_eol = request.get_json()["stop_on_double_eol"]
            if not isinstance(stop_on_double_eol, bool):
                return "stop_on_double_eol must be a boolean value"
        
        stop_on_eol = False
        if "stop_on_eol" in request.get_json():
            stop_on_eol = request.get_json()["stop_on_eol"]
            if not isinstance(stop_on_eol, bool):
                return "stop_on_eol must be a boolean value"

        random_seed = -1
        if "random_seed" in request.get_json():
            random_seed = request.get_json()["random_seed"]
            if not isinstance(random_seed, int):
                return "random_seed must be integer"
            if random_seed < 0: 
                return "random_seed must be a positive integer"

        no_log = False
        if "no_log" in request.get_json():
            no_log = request.get_json()["no_log"]
            if not isinstance(no_log, bool):
                return "no_log must be a boolean value"
        
        beam_width = None
        if "beam_width" in request.get_json():
            beam_width = request.get_json()["beam_width"]
            if not isinstance(beam_width, int):
                return "beam_width must be integer"
            if beam_width < 1:
                return "beam_width must be an integer > 1"
            if len(prompts) > 1:
                return "When doing beam_search, batch size must be 1"

        stop_token=7 # Default is <|endoftext|>
        if "stop_token" in request.get_json():
            stop_token = request.get_json()["stop_token"]
            if not isinstance(stop_token, int):
                return "stop_token must be an integer"
        
        length_penalty = 1 
        if "length_penalty" in request.get_json():
            length_penalty = request.get_json()["length_penalty"]
            if not isinstance(length_penalty, float):
                return "length_penalty must be a float"
        
        with lock:  # Need to get lock to keep multiple threads from hitting code
            
            if not no_log:
                # print(json.dumps(request.get_json()),flush=True)
                print("start time: ", datetime.datetime.now())
            
            try:
                do_generate(prompts, self.model, self.tokenizer, self.image_processor)
                # if beam_width is not None:
                #     mPLUGOwlGenerate.send_do_beam_search()  # Tell other ranks we're doing beam_search
                #     response, response_seg, response_scores = \
                #         beam_search_and_post_process(
                #         self.model,
                #         images=self.image_processor(images) if images else None,
                #         prompts=prompts,
                #         tokens_to_generate=tokens_to_generate,
                #         beam_size=beam_width,
                #         add_BOS=add_BOS,
                #         stop_token=stop_token,
                #         num_return_gen=beam_width,  # Returning whole beam
                #         length_penalty=length_penalty
                #         )
                #     # response = jsonify({"text": response,
                #     #     "segments": response_seg,
                #     #     "scores": response_scores})
                #     # response=[_[len(prompts[0]):] for i,_ in enumerate(response)]
                #     return response
                # else:
                #     mPLUGOwlGenerate.send_do_generate()  # Tell other ranks we're doing generate
                #     response, response_seg, response_logprobs = \
                #         generate_and_post_process(
                #         self.model,
                #         images=self.image_processor(images) if images else None,
                #         prompts=prompts,
                #         tokens_to_generate=tokens_to_generate,
                #         return_output_log_probs=logprobs,
                #         top_k_sampling=top_k,
                #         top_p_sampling=top_p,
                #         temperature=temperature,
                #         add_BOS=add_BOS,
                #         use_eod_token_for_early_termination=True,
                #         stop_on_double_eol=stop_on_double_eol,
                #         stop_on_eol=stop_on_eol,
                #         stop_token=stop_token,
                #         random_seed=random_seed)
                #     return response
                    # return response[0][len(prompts[0]):]
                    # return jsonify({"text": response,
                        # "segments": response[0][len(prompts[0]):],
                        # "logprobs": response_logprobs[0][len(prompts[0]):]})

            except ValueError as ve:
                import traceback
                traceback.print_exc()
                return "Length of prompt + tokens_to_generate longer than allowed"
                
            print("end time: ", datetime.datetime.now())
        

def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building XGPT3 model ...')
    model = XGPT3Model(
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process
    )
    return model


def add_text_generate_args(parser):
    group = parser.add_argument_group(title='text generation')

    group.add_argument("--temperature", type=float, default=1.0,
                       help='Sampling temperature.')
    group.add_argument("--top_p", type=float, default=0.0,
                       help='Top p sampling.')
    group.add_argument("--top_k", type=int, default=0,
                       help='Top k sampling.')
    group.add_argument("--out-seq-length", type=int, default=1024,
                       help='Size of the output generated text.')
    group.add_argument("--port", type=int, default=45678,
                       help='listening port')
    return parser


if __name__ == "__main__":
    model, tokenizer, img_processor = get_model(
            checkpoint_path='/nas-alinlp/qinghao.yqh/next_mplug/mplug_megatron_tp_v2/converted_lora.pth', tokenizer_path='/nas-alinlp/qinghao.yqh/ckpt/tokenizer.model')
    model.to(torch.bfloat16).cuda()
    model.eval()
    server = mPLUGOwlGenerate(model, tokenizer, img_processor)
    if mpu.is_pipeline_first_stage() and mpu.get_tensor_model_parallel_rank() == 0:
        
        
        CHAT_MODEL = server
        TOKENIZER = get_tokenizer()
        serve(app, host='0.0.0.0', port=args.port)
        print('> Inference Server Launched')

        # file_name = '/nas-alinlp/tjf141457/Megatron-v3/scripts/chat/server/test_entity_knowledge.search.jsonl'
        # # file_name = '/nas-alinlp/tjf141457/Megatron-v3/scripts/chat/server/test_inference'
        # output_data = []
        # with open(file_name, encoding='utf8') as f:
        #     cnt, all = 0, 0
        #     f.readline()
        #     for idx, line in enumerate(f):
        #         item = json.loads(line)['debug_info']
        #         prompt = item['fid_model_input']
                
        #         params = {
        #             "prompts": [prompt],
        #              "top_k":5, 
        #              "beam_width": 3,
        #              "stop_token": 1,
        #              "tokens_to_generate": 128
        #         }
        #         r = Request(params)
        #         response = server.inference(r)
        #         if isinstance(response,list):
        #             response = response[0]
        #         # add by yejiabo to remove words after the first stop token.
        #         # response = response.split('<|endoftext|>')[0]
        #         # response = response.split('</s>')[0]
        #         if 'ground_truth_answer' in item:
        #             ground_truth = item['ground_truth_answer']
        #             ground_truth = str(ground_truth).lower().split('|')
        #             for t in ground_truth:
        #                 if t in response.lower():
        #                     cnt += 1
        #                     break
        #         output_data.append({"prompt": prompt, "response": response})
        #         all += 1
        #         print(idx, response, cnt, all, cnt / all)
        # json.dump(output_data, open("answer_beam_search.json", "w", encoding="utf8"), indent=2, ensure_ascii=False) 
