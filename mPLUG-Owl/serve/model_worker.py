from PIL import Image
import torch
import gradio as gr
import logging
import sys
import os
import json
import requests
import datetime
import uuid
import base64
from io import BytesIO
import time
import sys
sys.path.append("..")

from .io_utils import IO, DefaultIO, OSS

import transformers
from mplug_owl.processing_mplug_owl import MplugOwlProcessor, MplugOwlImageProcessor
from mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl.configuration_mplug_owl import MplugOwlConfig
from transformers import GenerationConfig, AutoTokenizer

from .model_utils import post_process_output, Stream, Iteratorize

server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"

class mPLUG_Owl_Server:
    def __init__(
        self, 
        base_model='MAGAer13/mplug-owl-llama-7b',
        log_dir='./',
        load_in_8bit=False,
        bf16=True,
        device="cuda",
        io=None
    ):
        self.log_dir = log_dir
        self.image_processor = MplugOwlImageProcessor.from_pretrained(base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.processor = MplugOwlProcessor(self.image_processor, self.tokenizer)
        self.model = MplugOwlForConditionalGeneration.from_pretrained(
            base_model,
            load_in_8bit=load_in_8bit,
            torch_dtype=torch.bfloat16 if bf16 else torch.half,
            device_map="auto"
        )
        self.tokenizer = self.processor.tokenizer
        self.bf16 = bf16
        self.load_in_8bit = load_in_8bit

        if not load_in_8bit:
            if bf16:
                self.model.bfloat16()
            else:
                self.model.half()
        self.model.eval()

        self.io = io

    def evaluate(
        self, 
        pixel_values=None,
        input_ids=None,
        temperature=1.0,
        top_p=0.9,
        top_k=5,
        num_beams=3,
        max_new_tokens=256,
        stream_output=True,
        length_penalty=1.0,
        no_repeat_ngram_size=2,
        do_sample=False,
        early_stopping=True,
        **kwargs
    ):
        generation_config = dict(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            do_sample=do_sample,
            early_stopping=early_stopping,
            length_penalty=length_penalty,
        )
        
        generate_params = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }
        generate_params.update(generation_config)
        
        if stream_output:
            # Stream the reply 1 token at a time.
            # This is based on the trick of using 'stopping_criteria' to create an iterator,
            # from https://github.com/oobabooga/text-generation-webui/blob/ad37f396fc8bcbab90e11ecf17c56c97bfbd4a9c/modules/text_generation.py#L216-L243.

            def generate_with_callback(callback=None, **kwargs):
                kwargs.setdefault(
                    "stopping_criteria", transformers.StoppingCriteriaList()
                )
                kwargs["stopping_criteria"].append(Stream(callback_func=callback))
                with torch.no_grad():
                    self.model.generate(**kwargs)

            def generate_with_streaming(**kwargs):
                return Iteratorize(generate_with_callback, kwargs, callback=None)

            with generate_with_streaming(**generate_params) as generator:
                for output in generator:
                    # new_tokens = len(output) - len(input_ids[0])
                    decoded_output = self.tokenizer.decode(output)

                    if output[-1] in [self.tokenizer.eos_token_id]:
                        break

                    yield post_process_output(decoded_output)
            return  # early return for stream_output
        
        with torch.no_grad():
            generation_output = self.model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
                **generation_config
            )
        s = generation_output.sequences[0].cpu()
        output = self.tokenizer.decode(s)
        yield post_process_output(output)


    def predict(self, data):
        prompt = [data['text_input']]
        images = data['images'] if len(data['images']) > 0 else None
        if images:
            images = [Image.open(BytesIO(base64.b64decode(image))) for image in images]
        inputs = self.processor(text=prompt, images=images, return_tensors='pt')

        input_ids = inputs['input_ids'].to(self.model.device)
        if 'pixel_values' in inputs:
            if self.load_in_8bit:
                pixel_values = inputs['pixel_values'].half().to(self.model.device)
            elif self.bf16:
                pixel_values = inputs['pixel_values'].bfloat16().to(self.model.device)
            else:
                pixel_values = inputs['pixel_values'].half().to(self.model.device)
        else:
            pixel_values = None

        cache = None
        
        try:
            for x in self.evaluate(pixel_values, input_ids, stream_output=True, **data['generation_config']):
                cache = x
                yield (x, True)
        except ValueError as e:
            print("Caught ValueError:", e)
            yield (server_error_msg, False)
        except torch.cuda.CudaError as e:
            print("Caught torch.cuda.CudaError:", e)
            yield (server_error_msg, False)
        
        return
