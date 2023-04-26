from PIL import Image
# from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import gradio as gr
import logging
import sys
import os
import json
import requests
from .conversation import default_conversation
from .gradio_patch import Chatbot as grChatbot
from .gradio_css import code_highlight_css
import datetime
import uuid
import base64
from io import BytesIO
import time
from interface import do_generate

from .io_utils import IO, DefaultIO, OSS



handler = None


class _IOWrapper:
    def __init__(self):
        self._io = DefaultIO()

    def set_io(self, new_io):
        self._io = new_io

    def __getattr__(self, name):
        if hasattr(self._io, name):
            return getattr(self._io, name)
        return super().__getattr__(name)

    def __str__(self):
        return self._io.__name__

def init():
    global io

    io = _IOWrapper()


class mplug_owl:
    def __init__(self, device, checkpoint_path, tokenizer_path):
        self.device = device
        from interface import get_model
        model, tokenizer, img_processor = get_model(
                checkpoint_path=checkpoint_path, tokenizer_path=tokenizer_path)
        # model, tokenizer, img_processor = None, None, None
        self.model = model

        self.tokenizer = tokenizer
        self.img_processor = img_processor
        
        global io
        self.io = io
    
    def prediction(self, data, log_dir):

        now = datetime.datetime.now()
        today = now.strftime('%Y-%m-%d')
        if os.path.exists(os.path.join(log_dir, "image", today)) == False:
            self.io.makedirs(os.path.join(log_dir, "image", today))
            self.io.makedirs(os.path.join(log_dir, "chat", today))
        
        random_uuid = uuid.uuid4()
        random_uuid_str = str(random_uuid)

        log = {"chat": data["text_input"], "images": data["images"]}
        # if 'images' in data and len(data['images']) > 0:
        #     with self.io.open(os.path.join(log_dir, "image", today) + "/" + random_uuid_str+'.jpeg', 'wb') as f:
        #         # f.write(base64.decodebytes(data["images"][-1]))
        #         buffer = BytesIO(base64.b64decode(data["images"][-1]))
        #         f.write(buffer.getvalue())

        # generated_text = process(data, self.generator, self.tokenizer)
        generate_config = data.get('generate_config',
            {
                "top_k": 5, 
                "max_length": 512,
                "do_sample":True
            })
        if isinstance(generate_config,str):
            generate_config = json.loads(generate_config)
        if 'tokens_to_generate' in generate_config:
            generate_config['max_length'] = generate_config.pop('tokens_to_generate')
     
        generated_text = do_generate([data["text_input"]], data["images"], self.model, self.tokenizer,
                               self.img_processor, **generate_config)

        log["chat"] += generated_text

        with self.io.open(os.path.join(log_dir, "chat", today) + "/" + random_uuid_str+'.json', 'w') as f:
            json.dump(log, f)

        return generated_text

def get_inputs(text_input, images, topk, max_new_tokens, temperature=1):
    inputs = {
        "text_input": text_input,
        "images": images,
        "generate_config": json.dumps({
            # "random_seed": int(random_seed),
            "temperature": temperature,
            "top_k": int(topk), 
            # "add_BOS": True,
            # "beam_width": 3,
            # "stop_token": 2,
            "tokens_to_generate": int(max_new_tokens),
        }, ensure_ascii=False)
    }
    return inputs

def load_demo(url_params, request: gr.Request):

    dropdown_update = gr.Dropdown.update(visible=True)
    if "model" in url_params:
        model = url_params["model"]
        if model in models:
            dropdown_update = gr.Dropdown.update(
                value=model, visible=True)

    state = default_conversation.copy()

    return (state,
            dropdown_update,
            gr.Chatbot.update(visible=True),
            gr.Textbox.update(visible=True),
            gr.Button.update(visible=True),
            gr.Row.update(visible=True),
            gr.Accordion.update(visible=True))

def load_demo_refresh_model_list(request: gr.Request):
    models = get_model_list()
    state = default_conversation.copy()
    return (state, gr.Dropdown.update(
               choices=models,
               value=models[0] if len(models) > 0 else ""),
            gr.Chatbot.update(visible=True),
            gr.Textbox.update(visible=True),
            gr.Button.update(visible=True),
            gr.Row.update(visible=True),
            gr.Accordion.update(visible=True))

def vote_last_response(state, vote_type, model_selector, request: gr.Request):
    global io

    now = datetime.datetime.now()
    today = now.strftime('%Y-%m-%d')
    dst_path = os.path.join("oss://mm-chatgpt/mplug_owl_demo/log/feedback", today)
    try:
        io.makedirs(dst_path)
    except:
        pass

    with io.open(os.path.join(dst_path, "feedback.json"), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "state": state.dict(),
            # "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")

def upvote_last_response(state, model_selector, request: gr.Request):
    vote_last_response(state, "upvote", model_selector, request)
    return ("",) + (disable_btn,) * 3

def downvote_last_response(state, model_selector, request: gr.Request):
    vote_last_response(state, "downvote", model_selector, request)
    return ("",) + (disable_btn,) * 3

def flag_last_response(state, model_selector, request: gr.Request):
    vote_last_response(state, "flag", model_selector, request)
    return ("",) + (disable_btn,) * 3

def regenerate(state, request: gr.Request):
    state.messages[-1][-1] = None
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 5

def clear_history(request: gr.Request):
    state = default_conversation.copy()
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 5

def add_text(state, text, image, request: gr.Request):
    if len(text) <= 0 and image is None:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "", None) + (no_change_btn,) * 5

    if image is not None:
        multimodal_msg = None
        if '<image>' not in text:
            text = text + '\n<image>'

        if multimodal_msg is not None:
            return (state, state.to_gradio_chatbot(), multimodal_msg, None) + (
                no_change_btn,) * 5
        text = (text, image)
    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 5

def after_process_image(prompt):
    prompt = prompt.replace("\n<image>", "<image>")
    pro_prompt = ""
    prompt = prompt.split("\n")
    for p in prompt:
        if p.count("<image>") > 0:
            pro_prompt += "Human: <image>\n"
            if p != "":
                pro_prompt += p.replace("<image>", "") + "\n"
        else:
            pro_prompt += p + "\n"
    return (pro_prompt[:-1]+" ").replace("\n Human", "\nHuman").replace("\n AI", "\nAI")

headers = {"User-Agent": "mPLUG-owl Client"}

no_change_btn = gr.Button.update()
enable_btn = gr.Button.update(interactive=True)
disable_btn = gr.Button.update(interactive=False)

get_window_url_params = """
function() {
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    console.log(url_params);
    return url_params;
    }
"""