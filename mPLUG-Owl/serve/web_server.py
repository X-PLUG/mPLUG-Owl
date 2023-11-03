import argparse
import datetime
import json
import os
import time
import torch

import gradio as gr
import requests

from .conversation import default_conversation
from .gradio_css import code_highlight_css
from .gradio_patch import Chatbot as grChatbot
from .serve_utils import (
    add_text, after_process_image, disable_btn, no_change_btn,
    downvote_last_response, enable_btn, flag_last_response,
    get_window_url_params, init, regenerate, upvote_last_response
)
from .model_worker import mPLUG_Owl_Server
from .model_utils import post_process_code

SHARED_UI_WARNING = f'''### [NOTE] You can duplicate and use it with a paid private GPU.
<a class="duplicate-button" style="display:inline-block" target="_blank" href="https://huggingface.co/spaces/MAGAer13/mPLUG-Owl?duplicate=true"><img style="margin-top:0;margin-bottom:0" src="https://huggingface.co/datasets/huggingface/badges/raw/main/duplicate-this-space-md.svg" alt="Duplicate Space"></a>
'''

def load_demo(url_params, request: gr.Request):

    dropdown_update = gr.Dropdown.update(visible=True)
    state = default_conversation.copy()

    return (state,
            dropdown_update,
            gr.Chatbot.update(visible=True),
            gr.Textbox.update(visible=True),
            gr.Button.update(visible=True),
            gr.Row.update(visible=True),
            gr.Accordion.update(visible=True))

def clear_history(request: gr.Request):
    state = default_conversation.copy()

    return (state, state.to_gradio_chatbot(), "", None, None) + (disable_btn,) * 5

def http_bot(state, max_output_tokens, temperature, top_k, top_p, 
            num_beams, no_repeat_ngram_size, length_penalty,
            do_sample, request: gr.Request):
    if state.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (state, state.to_gradio_chatbot()) + (no_change_btn,) * 5
        return

    prompt = after_process_image(state.get_prompt())
    images = state.get_images()

    data = {
        "text_input": prompt,
        "images": images if len(images) > 0 else [],
        "generation_config": {
            "top_k": int(top_k),
            "top_p": float(top_p),
            "num_beams": int(num_beams),
            "no_repeat_ngram_size": int(no_repeat_ngram_size),
            "length_penalty": float(length_penalty),
            "do_sample": bool(do_sample),
            "temperature": float(temperature),
            "max_new_tokens": min(int(max_output_tokens), 1536),
        }
    }

    state.messages[-1][-1] = "▌"
    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5

    try:
        for chunk in model.predict(data):
            if chunk:
                if chunk[1]:
                    output = chunk[0].strip()
                    output = post_process_code(output)
                    state.messages[-1][-1] = output + "▌"
                    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5
                else:
                    output = chunk[0].strip()
                    state.messages[-1][-1] = output
                    yield (state, state.to_gradio_chatbot()) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
                    return
                time.sleep(0.03)

    except requests.exceptions.RequestException as e:
        state.messages[-1][-1] = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
        yield (state, state.to_gradio_chatbot()) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
        return

    state.messages[-1][-1] = state.messages[-1][-1][:-1]
    yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 5


def add_text_http_bot(
    state, text, image, video, 
    max_output_tokens, temperature, top_k, top_p, 
    num_beams, no_repeat_ngram_size, length_penalty,
    do_sample, request: gr.Request):
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

    yield (state, state.to_gradio_chatbot(), "", None, None) + (disable_btn,) * 5

    if state.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (state, state.to_gradio_chatbot(), "", None, None) + (no_change_btn,) * 5
        return

    prompt = after_process_image(state.get_prompt())
    images = state.get_images()

    data = {
        "text_input": prompt,
        "images": images if len(images) > 0 else [],
        "generation_config": {
            "top_k": int(top_k),
            "top_p": float(top_p),
            "num_beams": int(num_beams),
            "no_repeat_ngram_size": int(no_repeat_ngram_size),
            "length_penalty": float(length_penalty),
            "do_sample": bool(do_sample),
            "temperature": float(temperature),
            "max_new_tokens": min(int(max_output_tokens), 1536),
        }
    }

    state.messages[-1][-1] = "▌"
    yield (state, state.to_gradio_chatbot(), "", None, None) + (disable_btn,) * 5

    try:
        for chunk in model.predict(data):
            if chunk:
                if chunk[1]:
                    output = chunk[0].strip()
                    output = post_process_code(output)
                    state.messages[-1][-1] = output + "▌"
                    yield (state, state.to_gradio_chatbot(), "", None, None) + (disable_btn,) * 5
                else:
                    output = chunk[0].strip()
                    state.messages[-1][-1] = output
                    yield (state, state.to_gradio_chatbot(), "", None, None) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
                    return
                time.sleep(0.03)

    except requests.exceptions.RequestException as e:
        state.messages[-1][-1] = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
        yield (state, state.to_gradio_chatbot(), "", None, None) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
        return

    state.messages[-1][-1] = state.messages[-1][-1][:-1]
    yield (state, state.to_gradio_chatbot(), "", None, None) + (enable_btn,) * 5


def regenerate_http_bot(state, 
    max_output_tokens, temperature, top_k, top_p, 
    num_beams, no_repeat_ngram_size, length_penalty,
    do_sample, request: gr.Request):
    state.messages[-1][-1] = None
    state.skip_next = False
    yield (state, state.to_gradio_chatbot(), "", None, None) + (disable_btn,) * 5

    prompt = after_process_image(state.get_prompt())
    images = state.get_images()

    data = {
        "text_input": prompt,
        "images": images if len(images) > 0 else [],
        "generation_config": {
            "top_k": int(top_k),
            "top_p": float(top_p),
            "num_beams": int(num_beams),
            "no_repeat_ngram_size": int(no_repeat_ngram_size),
            "length_penalty": float(length_penalty),
            "do_sample": bool(do_sample),
            "temperature": float(temperature),
            "max_new_tokens": min(int(max_output_tokens), 1536),
        }
    }

    state.messages[-1][-1] = "▌"
    yield (state, state.to_gradio_chatbot(), "", None, None) + (disable_btn,) * 5

    try:
        for chunk in model.predict(data):
            if chunk:
                if chunk[1]:
                    output = chunk[0].strip()
                    output = post_process_code(output)
                    state.messages[-1][-1] = output + "▌"
                    yield (state, state.to_gradio_chatbot(), "", None, None) + (disable_btn,) * 5
                else:
                    output = chunk[0].strip()
                    state.messages[-1][-1] = output
                    yield (state, state.to_gradio_chatbot(), "", None, None) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
                    return
                time.sleep(0.03)

    except requests.exceptions.RequestException as e:
        state.messages[-1][-1] = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
        yield (state, state.to_gradio_chatbot(), "", None, None) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
        return

    state.messages[-1][-1] = state.messages[-1][-1][:-1]
    yield (state, state.to_gradio_chatbot(), "", None, None) + (enable_btn,) * 5

# [![Star on GitHub](https://img.shields.io/github/stars/X-PLUG/mPLUG-Owl.svg?style=social)](https://github.com/X-PLUG/mPLUG-Owl/stargazers)
# **If you are facing ERROR, it might be Out-Of-Memory (OOM) issue due to the limited GPU memory, please refresh the page to restart.** Besides, we recommand you to duplicate the space with a single A10 GPU to have a better experience. Or you can visit our demo hosted on [Modelscope](https://www.modelscope.cn/studios/damo/mPLUG-Owl/summary) which is hosted on a V100 machine.

title_markdown = ("""
<h1 align="center"><a href="https://github.com/X-PLUG/mPLUG-Owl"><img src="https://s1.ax1x.com/2023/05/12/p9yGA0g.png", alt="mPLUG-Owl" border="0" style="margin: 0 auto; height: 200px;" /></a> </h1>

<h2 align="center"> mPLUG-Owl🦉: Modularization Empowers Large Language Models with Multimodality </h2>

<h5 align="center"> If you like our project, please give us a star ✨ on Github for latest update.  </h2>

<div align="center">
    <div style="display:flex; gap: 0.25rem;" align="center">
        <a href='https://github.com/X-PLUG/mPLUG-Owl'><img src='https://img.shields.io/badge/Github-Code-blue'></a>
        <a href="https://arxiv.org/abs/2304.14178"><img src="https://img.shields.io/badge/Arxiv-2304.14178-red"></a>
        <a href='https://github.com/X-PLUG/mPLUG-Owl/stargazers'><img src='https://img.shields.io/github/stars/X-PLUG/mPLUG-Owl.svg?style=social'></a>
    </div>
</div>

**Notice**: The output is generated by top-k sampling scheme and may involve some randomness. For multiple images and video, we cannot ensure it's performance since only image-text pairs are used during training. For Video inputs, we recommand use the video **less than 10 seconds**.
""")

tos_markdown = ("""
### Terms of use
By using this service, users are required to agree to the following terms:
The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. The service may collect user dialogue data for future research.
Please click the "Flag" button if you get any inappropriate answer! We will collect those to keep improving our moderator.
For an optimal experience, please use desktop computers for this demo, as mobile devices may compromise its quality.

**Copyright 2023 Alibaba DAMO Academy.**
""")

learn_more_markdown = ("""
### License
The service is a research preview intended for non-commercial use only, subject to the model [License](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md) of LLaMA, [Terms of Use](https://openai.com/policies/terms-of-use) of the data generated by OpenAI, and [Privacy Practices](https://chrome.google.com/webstore/detail/sharegpt-share-your-chatg/daiacboceoaocpibfodeljbdfacokfjb) of ShareGPT. Please contact us if you find any potential violation.
""")

css = code_highlight_css + """
pre {
    white-space: pre-wrap;       /* Since CSS 2.1 */
    white-space: -moz-pre-wrap;  /* Mozilla, since 1999 */
    white-space: -pre-wrap;      /* Opera 4-6 */
    white-space: -o-pre-wrap;    /* Opera 7 */
    word-wrap: break-word;       /* Internet Explorer 5.5+ */
}
"""

def build_demo():
    # with gr.Blocks(title="mPLUG-Owl🦉", theme=gr.themes.Base(), css=css) as demo:
    with gr.Blocks(title="mPLUG-Owl🦉", css=css) as demo:
        state = gr.State()
        gr.Markdown(SHARED_UI_WARNING)

        gr.Markdown(title_markdown)

        with gr.Row():
            with gr.Column(scale=3):

                imagebox = gr.Image(type="pil")
                videobox = gr.Video()

                with gr.Accordion("Parameters", open=True, visible=False) as parameter_row:
                    max_output_tokens = gr.Slider(minimum=0, maximum=1024, value=512, step=64, interactive=True, label="Max output tokens",)
                    temperature = gr.Slider(minimum=0, maximum=1, value=1, step=0.1, interactive=True, label="Temperature",)
                    top_k = gr.Slider(minimum=1, maximum=5, value=3, step=1, interactive=True, label="Top K",)
                    top_p = gr.Slider(minimum=0, maximum=1, value=0.9, step=0.1, interactive=True, label="Top p",)
                    length_penalty = gr.Slider(minimum=1, maximum=5, value=1, step=0.1, interactive=True, label="length_penalty",)
                    num_beams = gr.Slider(minimum=1, maximum=5, value=1, step=1, interactive=True, label="Beam Size",)
                    no_repeat_ngram_size = gr.Slider(minimum=1, maximum=5, value=2, step=1, interactive=True, label="no_repeat_ngram_size",)
                    do_sample = gr.Checkbox(interactive=True, value=True, label="do_sample")

                gr.Markdown(tos_markdown)

            with gr.Column(scale=6):
                chatbot = grChatbot(elem_id="chatbot", visible=False).style(height=1000)
                with gr.Row():
                    with gr.Column(scale=8):
                        textbox = gr.Textbox(show_label=False,
                            placeholder="Enter text and press ENTER", visible=False).style(container=False)
                    with gr.Column(scale=1, min_width=60):
                        submit_btn = gr.Button(value="Submit", visible=False)
                with gr.Row(visible=False) as button_row:
                    upvote_btn = gr.Button(value="👍  Upvote", interactive=False)
                    downvote_btn = gr.Button(value="👎  Downvote", interactive=False)
                    flag_btn = gr.Button(value="⚠️  Flag", interactive=False)
                    regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
                    clear_btn = gr.Button(value="🗑️  Clear history", interactive=False)

        gr.Examples(examples=[
            [f"examples/monday.jpg", "Explain why this meme is funny."],
            [f'examples/rap.jpeg', 'Can you write me a master rap song that rhymes very well based on this image?'],
            [f'examples/titanic.jpeg', 'What happened at the end of this movie?'],
            [f'examples/vga.jpeg', 'What is funny about this image? Describe it panel by panel.'],
            [f'examples/mug_ad.jpeg', 'We design new mugs shown in the image. Can you help us write an advertisement?'],
            [f'examples/laundry.jpeg', 'Why this happens and how to fix it?'],
            [f'examples/ca.jpeg', "What do you think about the person's behavior?"],
            [f'examples/monalisa-fun.jpg', 'Do you know who drew this painting?​'],
        ], inputs=[imagebox, textbox])

        gr.Markdown(learn_more_markdown)
        url_params = gr.JSON(visible=False)

        btn_list = [upvote_btn, downvote_btn, flag_btn, regenerate_btn, clear_btn]
        parameter_list = [
            max_output_tokens, temperature, top_k, top_p, 
            num_beams, no_repeat_ngram_size, length_penalty,
            do_sample
        ]
        upvote_btn.click(upvote_last_response,
            [state], [textbox, upvote_btn, downvote_btn, flag_btn])
        downvote_btn.click(downvote_last_response,
            [state], [textbox, upvote_btn, downvote_btn, flag_btn])
        flag_btn.click(flag_last_response,
            [state], [textbox, upvote_btn, downvote_btn, flag_btn])
        # regenerate_btn.click(regenerate, state,
        #     [state, chatbot, textbox, imagebox, videobox] + btn_list).then(
        #     http_bot, [state] + parameter_list,
        #     [state, chatbot] + btn_list)
        regenerate_btn.click(regenerate_http_bot, [state] + parameter_list,
            [state, chatbot, textbox, imagebox, videobox] + btn_list)

        clear_btn.click(clear_history, None, [state, chatbot, textbox, imagebox, videobox] + btn_list)

        # textbox.submit(add_text, [state, textbox, imagebox, videobox], [state, chatbot, textbox, imagebox, videobox] + btn_list
        #     ).then(http_bot, [state] + parameter_list,
        #            [state, chatbot] + btn_list)
        # submit_btn.click(add_text, [state, textbox, imagebox, videobox], [state, chatbot, textbox, imagebox, videobox] + btn_list
        #     ).then(http_bot, [state] + parameter_list,
        #            [state, chatbot] + btn_list)

        textbox.submit(add_text_http_bot, 
            [state, textbox, imagebox, videobox] + parameter_list, 
            [state, chatbot, textbox, imagebox, videobox] + btn_list
        )

        submit_btn.click(add_text_http_bot, 
            [state, textbox, imagebox, videobox] + parameter_list, 
            [state, chatbot, textbox, imagebox, videobox] + btn_list
        )

        demo.load(load_demo, [url_params], [state,
            chatbot, textbox, submit_btn, button_row, parameter_row],
            _js=get_window_url_params)

    return demo

if __name__ == "__main__":
    io = init()
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = cur_dir[:-9] + "log"

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--debug", action="store_true", help="using debug mode")
    parser.add_argument("--port", type=int)
    parser.add_argument("--concurrency-count", type=int, default=100)
    parser.add_argument("--base-model",type=str, default='MAGAer13/mplug-owl-llama-7b')
    parser.add_argument("--load-8bit", action="store_true", help="using 8bit mode")
    parser.add_argument("--bf16", action="store_true", help="using 8bit mode")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model = mPLUG_Owl_Server(
        base_model=args.base_model,
        log_dir=log_dir,
        load_in_8bit=args.load_8bit,
        bf16=args.bf16,
        device=device,
        io=io
    )
    demo = build_demo()
    demo.queue(concurrency_count=args.concurrency_count, status_update_rate=10, api_open=False).launch(server_name=args.host, debug=args.debug, server_port=args.port, share=False)

