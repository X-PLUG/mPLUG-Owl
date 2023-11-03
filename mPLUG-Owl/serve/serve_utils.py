from PIL import Image
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
    io = _IOWrapper()
    return io


def vote_last_response(state, vote_type, model_selector, request: gr.Request):
    pass

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
    return (state, state.to_gradio_chatbot(), "", None, None) + (disable_btn,) * 5

def clear_history(request: gr.Request):
    state = default_conversation.copy()
    return (state, state.to_gradio_chatbot(), "", None, None) + (disable_btn,) * 5


def add_text(state, text, image, video, request: gr.Request):
    if len(text) <= 0 and (image is None or video is None):
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "", None, None) + (no_change_btn,) * 5

    if image is not None:
        if '<image>' not in text:
            text = text + '\n<image>'
        text = (text, image)
    
    if video is not None:
        num_frames = 4
        if '<image>' not in text:
            text = text + '\n<image>' * num_frames
        text = (text, video)

    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "", None, None) + (disable_btn,) * 5

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


headers = {"User-Agent": "mPLUG-Owl Client"}

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