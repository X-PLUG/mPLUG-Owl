import torch
from torchvision import transforms
from PIL import Image
import random

from data_utils.randaugment import RandomAugment
from .builder import PROCESSORS


@PROCESSORS.register_module()
class CaptionProcessor:
    def __init__(self, image_size=224, min_scale = 0.5, randaug=False):
        self.image_size = image_size
        self.min_scale = min_scale

        if randaug:
            self.image_transform = transforms.Compose([                        
                transforms.RandomResizedCrop(image_size,scale=(min_scale, 1.0), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                                'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        else:
            self.image_transform = transforms.Compose([                        
                transforms.RandomResizedCrop(image_size,scale=(min_scale, 1.0), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),  
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