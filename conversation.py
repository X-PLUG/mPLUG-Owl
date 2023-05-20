import dataclasses
from enum import auto, Enum
from typing import List, Tuple
import os
from decord import VideoReader
import numpy as np
from PIL import Image

class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()

@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "\n "
    sep2: str = None

    skip_next: bool = False

    def get_prompt(self):
        self.system = "The following is a conversation between a curious human and AI. The AI gives helpful, detailed, and polite answers to the human's questions."
        self.sep = "\n"
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    if type(message) is tuple:
                        message, _ = message
                    ret += role.replace("AI", "AI") + ": " + message + self.sep
                else:
                    if role != "":
                        ret += role.replace("AI", "AI") + ":"
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if type(message) is tuple:
                        message, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def get_index(self, num_frames, num_segments):
        seg_size = float(num_frames - 1) / num_segments
        start = int(seg_size / 2)
        offsets = np.array([
            start + int(np.round(seg_size * idx)) for idx in range(num_segments)
        ])
        return offsets

    def load_video(self, path, num_frames=4):
        vr = VideoReader(path, height=224, width=224)
        total_frames = len(vr)
        frame_indices = self.get_index(total_frames, num_frames)
        images_group = list()
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
            images_group.append(img)
        return images_group

    def get_images(self, log_dir=None):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        images = []
        k = 0
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    import base64
                    from io import BytesIO
                    msg, image = msg
                    image_tmp = image
                    if isinstance(image_tmp, str):
                        image_pils = self.load_video(image_tmp)
                    else:
                        image_pils = [image_tmp]

                    for image in image_pils:
                        buffered = BytesIO()
                        
                        image.save(buffered, format="JPEG")
                        
                        img_str = base64.b64encode(buffered.getvalue()).decode()
                        images.append(img_str)
                    k += 1
        return images
    
    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    import base64
                    from io import BytesIO
                    msg, image = msg
                    if isinstance(image, str):
                        with open(image, 'rb') as f:
                            data = f.read()
                        img_b64_str = base64.b64encode(data).decode()
                        image_str = f'<video src="data:video/mp4;base64,{img_b64_str}" controls width="426" height="240"></video>'
                        msg = msg.replace('\n'.join(['<image>']*4), image_str)
                    else:
                        max_hw, min_hw = max(image.size), min(image.size)
                        aspect_ratio = max_hw / min_hw
                        max_len, min_len = 800, 400
                        shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
                        longest_edge = int(shortest_edge * aspect_ratio)
                        W, H = image.size
                        if H > W:
                            H, W = longest_edge, shortest_edge
                        else:
                            H, W = shortest_edge, longest_edge
                        image = image.resize((W, H))
                        # image = image.resize((224, 224))
                        buffered = BytesIO()
                        image.save(buffered, format="JPEG")
                        img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                        img_str = f'<img src="data:image/png;base64,{img_b64_str}" alt="user upload image" />'
                        msg = msg.replace('<image>', img_str)
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret
    
    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2)

    def dict(self):
        if len(self.get_images()) > 0:
            return {
                "system": self.system,
                "roles": self.roles,
                "messages": [[x, y[0] if type(y) is tuple else y] for x, y in self.messages],
                "offset": self.offset,
                "images": self.get_images(),
                "sep": self.sep,
                "sep2": self.sep2,
            }
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }
        
mplug_owl_v0 = Conversation(
    system="The following is a conversation between a curious human and assistant AI. The assistant AI gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "AI"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

default_conversation = mplug_owl_v0

if __name__ == "__main__":
    print(default_conversation.get_prompt())
