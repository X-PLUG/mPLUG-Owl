import torch
from torchvision import transforms
from PIL import Image
import random

from data_utils.randaugment import RandomAugment
from .builder import PROCESSORS


@PROCESSORS.register_module()
class DefaultProcessor:
    def __init__(self, image_size=224):
        self.image_size = image_size

        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size),interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        self.text_transform = None

    def __call__(self, image, text):
        assert image or text

        if image:
            image_input = self.image_transform(image)
        else:
            image_input = None

        if text:
            if isinstance(text["prompt"], list):
                prompt = random.choice(text["prompt"])
            else:
                prompt = text["prompt"]
            text_input = dict(
                prompt=prompt,
                completion=text["text"],
            )
        else:
            text_input = None
        return image_input, text_input