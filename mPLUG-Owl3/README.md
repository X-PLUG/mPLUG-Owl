<div style="display: flex; align-items: center;">
  <h1>mPLUG-Owl3: Towards Long Image-Sequence Understanding in Multi-Modal Large Language Models</h1>
</div>

<div align="center">
<img src="assets/mplug_owl2_logo.png" width="20%">
</div>

<div align="center">
Jiabo Ye, Haiyang Xu, Haowei Liu, Anwen Hu, Ming Yan, Qi Qian, Ji Zhang, Fei Huang, Jingren Zhou
</div>
<div align="center">
<strong>Tongyi, Alibaba Group</strong>
</div>
<div align="center">
</div>


<div align="center">
    <a href="https://github.com/X-PLUG/mPLUG-Owl/blob/main/LICENSE"><img src="assets/LICENSE-Apache%20License-blue.svg" alt="License"></a>
    <a href="https://arxiv.org/abs/2408.04840"><img src="assets/Paper-Arxiv-orange.svg" ></a>
    <a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FX-PLUG%2FmPLUG-Owl&count_bg=%23E97EBA&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=visitors&edge_flat=false" alt="Hits"></a>

</div>

---
**mPLUG-Owl3** 
<div align="center">
</div>

![Performance and Efficiency](./assets/performance.png)

## News and Updates
* ```2024.08.12``` 🔥🔥🔥 We release **mPLUG-Owl3**. The source code and weights are avaliable at [HuggingFace](https://huggingface.co/mPLUG/mPLUG-Owl3-7B-240728).

## Cases
mPLUG-Owl3 can learn from knowledge from retrieval system.

![RAG ability](./assets/fewshot.png)

mPLUG-Owl3 can also chat with user with a interleaved image-text context.

![Interleaved image-text Dialogue](./assets/multiturn.png)

mPLUG-Owl3 can watch long videos such as movies and remember its details. 

![Long video understanding](./assets/movie.png)
## TODO List
- [ ] Evaluation with huggingface model.
- [ ] Training pipeline.

## Performance
Visual Question Answering
![VQA](./assets/vqa.png)
Multimodal LLM Benchmarks
![Multimodal Benchmarks](./assets/mmb.png)
Video Benchmarks
![Video Benchmarks](./assets/video_bench.png)
Multi-image Benchmarks
![Multiimage Benchmarks](./assets/multiimage_bench.png)
![MI-Bench](./assets/mibench.png)

## Checkpoints
[HuggingFace](https://huggingface.co/mPLUG/mPLUG-Owl3-7B-240728)

## Usage

## Quickstart

Load the mPLUG-Owl3. We now only support attn_implementation in ```['sdpa', 'flash_attention_2']```.
```Python
import torch
model_path = 'mPLUG/mPLUG-Owl3-7B-240728'
config = mPLUGOwl3Config.from_pretrained(model_path)
print(config)
# model = mPLUGOwl3Model(config).cuda().half()
model = mPLUGOwl3Model.from_pretrained(model_path, attn_implementation='sdpa', torch_dtype=torch.half)
model.eval().cuda()
```
Chat with images.
```Python
from PIL import Image

from transformers import AutoTokenizer, AutoProcessor
from decord import VideoReader, cpu    # pip install decord
model_path = 'mPLUG/mPLUG-Owl3-7B-240728'
tokenizer = AutoTokenizer.from_pretrained(model_path)
processor = model.init_processor(tokenizer)

image = Image.new('RGB', (500, 500), color='red')

messages = [
    {"role": "user", "content": """<|image|>
Describe this image."""},
    {"role": "assistant", "content": ""}
]

inputs = processor(messages, images=image, videos=None)

inputs.to('cuda')
inputs.update({
    'tokenizer': tokenizer,
    'max_new_tokens':100,
    'decode_text':True,
})


g = model.generate(**inputs)
print(g)
```

Chat with a video.
```Python
from PIL import Image

from transformers import AutoTokenizer, AutoProcessor
from decord import VideoReader, cpu    # pip install decord
model_path = 'mPLUG/mPLUG-Owl3-7B-240728'
tokenizer = AutoTokenizer.from_pretrained(model_path)
processor = model.init_processor(tokenizer)


messages = [
    {"role": "user", "content": """<|video|>
Describe this video."""},
    {"role": "assistant", "content": ""}
]

videos = ['/nas-mmu-data/examples/car_room.mp4']

MAX_NUM_FRAMES=16

def encode_video(video_path):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    print('num frames:', len(frames))
    return frames
video_frames = [encode_video(_) for _ in videos]
inputs = processor(messages, images=None, videos=video_frames)

inputs.to('cuda')
inputs.update({
    'tokenizer': tokenizer,
    'max_new_tokens':100,
    'decode_text':True,
})


g = model.generate(**inputs)
print(g)
```


## Citation

If you find mPLUG-Owl3 useful for your research and applications, please cite using this BibTeX:
```bibtex
@misc{ye2024mplugowl3longimagesequenceunderstanding,
      title={mPLUG-Owl3: Towards Long Image-Sequence Understanding in Multi-Modal Large Language Models},
      author={Jiabo Ye and Haiyang Xu and Haowei Liu and Anwen Hu and Ming Yan and Qi Qian and Ji Zhang and Fei Huang and Jingren Zhou},
      year={2024},
      eprint={2408.04840},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.04840},
}
```

## Acknowledgement

- [LLaVA](https://github.com/haotian-liu/LLaVA): the codebase we built upon. Thanks for the authors of LLaVA for providing the framework.


## Related Projects

* [LLaMA](https://github.com/facebookresearch/llama). A open-source collection of state-of-the-art large pre-trained language models.
* [LLaVA](https://github.com/haotian-liu/LLaVA). A visual instruction tuned vision language model which achieves GPT4 level capabilities.
* [mPLUG](https://github.com/alibaba/AliceMind/tree/main/mPLUG). A vision-language foundation model for both cross-modal understanding and generation.
* [mPLUG-2](https://github.com/alibaba/AliceMind). A multimodal model with a modular design, which inspired our project.
