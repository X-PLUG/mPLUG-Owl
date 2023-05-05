<div align="center">
<img src="assets/mPLUG_new1.png" width="80%">
</div>

# mPLUG-Owl🦉: Modularization Empowers Large Language Models with Multimodality
<div align="center">
Qinghao Ye*, Haiyang Xu*, Guohai Xu*, Jiabo Ye, Ming Yan†, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, Chaoya Jiang, Chenliang Li, Yuanhong Xu, Hehong Chen, Junfeng Tian, Qian Qi, Ji Zhang, Fei Huang
</div>

<div align="center">
<strong>阿里巴巴集团，达摩院</strong>
</div>

<div align="center">
*Equal Contribution; † Corresponding Author
</div>

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm-dark.svg)](https://huggingface.co/spaces/MAGAer13/mPLUG-Owl)
[![](assets/Demo-ModelScope-brightgreen.svg)](https://modelscope.cn/studios/damo/mPLUG-Owl/summary)
[![](assets/LICENSE-Apache%20License-blue.svg)](https://github.com/X-PLUG/mPLUG-Owl/blob/main/LICENSE)
[![](assets/Paper-PDF-orange.svg)](http://mm-chatgpt.oss-cn-zhangjiakou.aliyuncs.com/mplug_owl_demo/released_checkpoint/mPLUG_Owl_paper.pdf)
[![](assets/Paper-Arxiv-orange.svg)](https://arxiv.org/abs/2304.14178)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FX-PLUG%2FmPLUG-Owl&count_bg=%23E97EBA&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=visitors&edge_flat=false)](https://hits.seeyoufarm.com)

[English](README.md) | 简体中文
<hr>

<div align="center">
<img src="http://mm-chatgpt.oss-cn-zhangjiakou.aliyuncs.com/mplug_owl_demo/released_checkpoint/sample.gif"  width="60%">
</div>

## 示例
![Training paradigm and model overview](assets/case_1.png "Training paradigm and model overview")
![Training paradigm and model overview](assets/case_2.png "Training paradigm and model overview")

## 最新更新

* 我们发布了指令微调的代码。
* 我们在[HuggingFace](https://huggingface.co/spaces/MAGAer13/mPLUG-Owl)上也搭建了Demo。感谢HuggingFace提供的免费算力！
* HuggingFace上的Demo现在已经支持视频输入！ModelScope上的Demo也即将支持。
* 我们公开了视觉相关指令的测评集**OwlEval**
* 我们在Modelscope上提供了一个[在线Demo](https://modelscope.cn/studios/damo/mPLUG-Owl/summary)供大家体验。
* 我们开放了mPLUG-Owl🦉，以及推理代码和二阶段微调参数。

## 亮点特色
* 一种面向多模态语言模型的**模块化**的训练范式。
* 能学习与语言空间相适应的视觉知识，并支持在多模态场景下进行**多轮对话**。
* 涌现**多图关系理解**，**场景文本理解**和**基于视觉的文档理解**等能力。
* 提出了针对视觉相关指令的测评集**OwlEval**，用以评估多模态语言模型的对带有视觉信息上下文的理解能力。
* 我们在模块化上的一些探索:
  * [E2E-VLP](https://aclanthology.org/2021.acl-long.42/), [mPLUG](https://aclanthology.org/2022.emnlp-main.488/) 和 [mPLUG-2](https://arxiv.org/abs/2302.00402), 分别被ACL 2021, EMNLP 2022 and ICML 2023接收。
  * [mPLUG](https://aclanthology.org/2022.emnlp-main.488/) 首次在VQA上超越人类。
* 即将发布
  - [ ] 在HuggingFace Hub上发布。
  - [ ] 多语言支持（中文、日文等）。
  - [ ] 在多图片/视频数据上训练的模型
  - [x] Huggingface 在线Demo
  - [x] 指令微调代码。
  - [x] 视觉相关指令的测评集**OwlEval**

![Training paradigm and model overview](assets/model.png "Training paradigm and model overview")

## 在线Demo
### ModelScope
<a href="https://www.modelscope.cn/studios/damo/mPLUG-Owl/summary"><img src="https://modelscope.oss-cn-beijing.aliyuncs.com/modelscope.gif" width="250"/></a>

### Hugging Face
<!-- [![Demo of mPLUG-Owl on Modelscope](assets/modelscopeIcon.svg)](https://www.modelscope.cn/studios/damo/mPLUG-Owl/summary) -->

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-xl-dark.svg)](https://huggingface.co/spaces/MAGAer13/mPLUG-Owl)
![](assets/modelscope.png)
## 预训练参数
|Model|Phase|Download link|
|-|-|-|
|mPLUG-Owl 7B|Pre-training|[下载链接](http://mm-chatgpt.oss-cn-zhangjiakou.aliyuncs.com/mplug_owl_demo/released_checkpoint/pretrained.pth)|
|mPLUG-Owl 7B|Instruction tuning|[下载链接](http://mm-chatgpt.oss-cn-zhangjiakou.aliyuncs.com/mplug_owl_demo/released_checkpoint/instruction_tuned.pth)|
|Tokenizer model|N/A|[下载链接](http://mm-chatgpt.oss-cn-zhangjiakou.aliyuncs.com/mplug_owl_demo/released_checkpoint/tokenizer.model)|

## OwlEval
我们所使用的评测集放在 ```./OwlEval``` 中。

## 使用
### 安装依赖
核心组件:
* PyTorch=1.13.1
* transformers=4.28.1
* [Apex](https://github.com/NVIDIA/apex)
* einops
* icecream
* flask
* ruamel.yaml
* uvicorn 
* fastapi
* markdown2
* gradio
* sconf
* h5py
* sentencepiece
* peft

你也可以根据我们导出的```env.yaml```来准备你的环境。

Apex需要手动从其源码进行编译，因为mPLUG-Owl依赖它的cpp extension (MixedFusedLayerNorm)。

考虑到apex仓库的代码会频繁变动，我们在本仓库中内嵌了一个固定的apex源码（验证过了），你可以通过下面的命令进行安装：
```shell
cd apex_22.01_pp
TORCH_CUDA_ARCH_LIST='5.2 6.0 6.1 7.0 7.5 8.0 8.6' pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

下个版本，我们会移除掉对apex的依赖。

### 本地部署Demo
我们提供了一个易扩展的脚本来一键部署本地Demo，你可以根据自己的需求进行修改。
```Bash
python -m server_mplug.owl_demo --debug --port 6363 --checkpoint_path 'your checkpoint path' --tokenizer_path 'your tokenizer path'
```
### 推理
如果要实现自定义的推理，可以参考以下步骤。

构建model, tokenizer, img_processor
```Python
from interface import get_model
model, tokenizer, img_processor = get_model(
        checkpoint_path='checkpoint path', tokenizer_path='tokenizer path')
```
准备模型输入
```Python
# We use a human/AI template to organize the context as a multi-turn conversation.
# <image> denotes an image placehold.
prompts = [
'''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Human: <image>
Human: Explain why this meme is funny.
AI: ''']

# The image paths should be placed in the image_list and kept in the same order as in the prompts.
# We support urls, local file paths and base64 string. You can custom the pre-process of images by modifying the mplug_owl.modeling_mplug_owl.ImageProcessor
image_list = ['https://xxx.com/image.jpg',]
```

对于多张图片，由于这是模型涌现的能力（训练数据并不包含多图的样本），我们并不知道何种格式对模型而言是最佳的。下面是我们在实验中所使用的格式。大家也可以自行探索能让模型更好理解多图关系的格式或者prompt。
```Python
prompts = [
'''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Human: <image>
Human: <image>
Human: Do the shirts worn by the individuals in the first and second pictures vary in color? If so, what is the specific color of each shirt?
AI: ''']

image_list = ['https://xxx.com/image_1.jpg', 'https://xxx.com/image_2.jpg']
```

获取模型回复
```Python
# generate kwargs (the same in transformers) can be passed in the do_generate()
from interface import do_generate
sentence = do_generate(prompts, image_list, model, tokenizer,
                               img_processor, max_length=512, top_k=5, do_sample=True)
```
### 指令微调
训练样本存放在```xxx.jsonl``` 中，格式如下:
```json
{"text": "The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nHuman: Construct a sentence using the given verb in the past tense\nshudder\nAI: She shuddered at the thought of being alone in the dark.", "task_type": "gpt4instruct_sft"}
{"image": ["xxx.jpg"], "text": "The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nHuman: <image>\nHuman: Where is the man located in the image?\nAI: The man is located in a small restaurant, sitting and enjoying his meal.\nHuman: What type of food is the man eating in the image?\nAI: The man is eating a burrito in the image.\nHuman: What can be seen next to the man in the restaurant?\nAI: Next to the man in the restaurant, there is a fridge filled with cans of soda.\nHuman: How does the man feel as he is about to eat his meal?\nAI: The man looks happy and is smiling as he is about to take a bite of his meal.\nHuman: What unique aspect of the photo makes the scene appear distorted or unusual?\nAI: The unique aspect of this photo is that it was taken using a fish-eye lens, which creates a distorted, wide-angle view of the scene, making it appear different from a standard perspective.", "task_type": "llava_sft"}
```
其中 ```task_type``` 可以为 ```{'quora_chat_sft', 'sharegpt_chat_sft', 'llava_sft', 'gpt4instruct_sft'}```。

准备好 train.jsonl 和 dev.jsonl 并修改 ```configs/instruction_tuning/v0.yaml``` 中的 ```data_files```.

执行训练脚本。
```
bash train_it.sh
```
## 性能比较
我们展示了50个单轮对话（左）和52个多轮对话（右）在人工评估指标下，mPLUG-Owl和基线方法的比较结果。A/B/C/D表示评分人员对每个回复的评级。

![Comparison Results](assets/mPLUG_Owl_compare_result_s&mturn.png)

## 相关项目

* [LLaMA](https://github.com/facebookresearch/llama). 开源的大型预训练语言模型系列。
* [Baize](https://github.com/project-baize/baize-chatbot). 使用LoRA在10万个通过让ChatGPT自聊生成的对话上进行训练的开源聊天模型。
* [Alpaca](https://github.com/tatsu-lab/stanford_alpaca). 从7B LLaMA模型上进行微调训练的，用于52K个指令数据的模型。
* [LoRA](https://github.com/microsoft/LoRA). 即插即用的模块，可以极大地减少下游任务的可训练参数数量。
* [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4). 仅使用一个线性映射层，将冻结的语言模型和视觉编码器结合的多模态语言模型。
* [LLaVA](https://github.com/haotian-liu/LLaVA). 经过视觉指令调整的视觉语言模型，可以实现GPT4级别的能力。
* [mPLUG](https://github.com/alibaba/AliceMind/tree/main/mPLUG). 视觉语言基础模型，可以用于跨模态理解和生成。
* [mPLUG-2](https://github.com/alibaba/AliceMind). 具有模块化设计的多模态模型，启发了我们的项目。

## 引用
如果我们的工作对你有帮助，可以考虑给我们的仓库点个star & 引用我们的论文。
```
@misc{ye2023mplugowl,
      title={mPLUG-Owl: Modularization Empowers Large Language Models with Multimodality}, 
      author={Qinghao Ye and Haiyang Xu and Guohai Xu and Jiabo Ye and Ming Yan and Yiyang Zhou and Junyang Wang and Anwen Hu and Pengcheng Shi and Yaya Shi and Chaoya Jiang and Chenliang Li and Yuanhong Xu and Hehong Chen and Junfeng Tian and Qian Qi and Ji Zhang and Fei Huang},
      year={2023},
      eprint={2304.14178},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
