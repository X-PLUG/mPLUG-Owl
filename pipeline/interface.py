import torch
import numpy as np
import requests
from PIL import Image
from mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor
from transformers import AutoTokenizer


def get_model(pretrained_ckpt, use_bf16=False):
    """Model Provider with tokenizer and processor. 

    Args:
        pretrained_ckpt (string): The path to pre-trained checkpoint.
        use_bf16 (bool, optional): Whether to use bfloat16 to load the model. Defaults to False.

    Returns:
        model: MplugOwl Model
        tokenizer: MplugOwl text tokenizer
        processor: MplugOwl processor (including text and image)
    """
    model = MplugOwlForConditionalGeneration.from_pretrained(
        pretrained_ckpt,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.half,
    )
    image_processor = MplugOwlImageProcessor.from_pretrained(pretrained_ckpt)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_ckpt)
    processor = MplugOwlProcessor(image_processor, tokenizer)
    return model, tokenizer, processor


def do_generate(prompts, image_list, model, tokenizer, processor, use_bf16=False, **generate_kwargs):
    """The interface for generation

    Args:
        prompts (List[str]): The prompt text
        image_list (List[str]): Paths of images
        model (MplugOwlForConditionalGeneration): MplugOwlForConditionalGeneration
        tokenizer (AutoTokenizer): AutoTokenizer
        processor (MplugOwlProcessor): MplugOwlProcessor
        use_bf16 (bool, optional): Whether to use bfloat16. Defaults to False.

    Returns:
        sentence (str): Generated sentence.
    """
    if image_list:
        images = [Image.open(_) for _ in image_list]
    else:
        images = None
    inputs = processor(text=prompts, images=images, return_tensors='pt')
    inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        res = model.generate(**inputs, **generate_kwargs)
    sentence = tokenizer.decode(res.tolist()[0], skip_special_tokens=True)
    return sentence


if __name__ == '__main__':
    prompts = ['''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Human: <image>
Human: Explain why this meme is funny.
AI: ''']
    image_list = ['examples/monday.jpg']

    # base_model = 'MAGAer13/mplug-owl-llama-7b'
    base_model = 'output/sft_v0.1_lora_grad_ckpt/checkpoint-4000/generation_config.json'
    model, tokenizer, processor = get_model(base_model, use_bf16=True)
    sentence = do_generate(
        prompts, image_list, model, 
        tokenizer, processor, use_bf16=True,
        max_length=512, top_k=5, do_sample=True
    )
    print(sentence)
  