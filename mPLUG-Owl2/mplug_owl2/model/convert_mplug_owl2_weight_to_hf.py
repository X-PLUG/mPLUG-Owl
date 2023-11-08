# Copyright 2023 DAMO Academy and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import gc
import json
import math
import os
import shutil
import warnings

import torch

from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer
from .configuration_mplug_owl2 import MPLUGOwl2Config, MplugOwlVisionConfig, MplugOwlVisualAbstractorConfig
from .modeling_mplug_owl2 import MPLUGOwl2LlamaForCausalLM

try:
    from transformers import LlamaTokenizerFast
except ImportError as e:
    warnings.warn(e)
    warnings.warn(
        "The converted tokenizer will be the `slow` tokenizer. To use the fast, update your `tokenizers` library and re-run the tokenizer conversion"
    )
    LlamaTokenizerFast = None

"""
Sample usage:

```
python3 /pure-mlo-scratch/sfan/model-parallel-trainer/llama2megatron/convert_llama2hf.py \
    --input_dir /pure-mlo-scratch/llama/ --model_size 7 --output_dir /pure-mlo-scratch/llama/converted_HF_7B
```

Thereafter, models can be loaded via:

```py
from transformers import LlamaForCausalLM, LlamaTokenizer

model = LlamaForCausalLM.from_pretrained("/output/path")
tokenizer = LlamaTokenizer.from_pretrained("/output/path")
```

Important note: you need to be able to host the whole model in RAM to execute this script (even if the biggest versions
come in several checkpoints they each contain a part of each weight of the model, so we need to load them all in RAM).
"""

llama_s2layer = {7: 32, 13: 40, 30: 60, 65: 80, 70: 80}
llama_s2heads = {7: 32, 13: 40, 30: 52, 65: 64, 70: 64}
llama_s2dense = {7: 11008, 13: 13824, 30: 17920, 65: 22016,
                 70: 28672}  # should be (2/3)*4*d, but it isn't exaclty that
llama_s2hidden = {7: 4096, 13: 5120, 32: 6656, 65: 8192, 70: 8192}


def compute_intermediate_size(n):
    return int(math.ceil(n * 8 / 3) + 255) // 256 * 256


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f)


def write_model(model_path, 
                input_base_path, 
                model_size,
                num_input_shards=1,
                num_output_shards=2,
                skip_permute=True,
                norm_eps=1e-05):
    # if os.path.exists(model_path):
    #     shutil.rmtree(model_path)
    os.makedirs(model_path, exist_ok=True)
    # tmp_model_path = os.path.join(model_path, "tmp")
    tmp_model_path = model_path
    os.makedirs(tmp_model_path, exist_ok=True)

    num_shards = num_input_shards
    n_layers = llama_s2layer[model_size]
    n_heads = llama_s2heads[model_size]
    n_heads_per_shard = n_heads // num_shards
    n_dense = llama_s2dense[model_size]
    n_hidden = llama_s2hidden[model_size]
    hidden_per_head = n_hidden // n_heads
    base = 10000.0
    inv_freq = 1.0 / (base ** (torch.arange(0, hidden_per_head, 2).float() / hidden_per_head))

    # permute for sliced rotary
    def permute(w, skip_permute=skip_permute):
        if skip_permute:
            return w
        return w.view(n_heads, n_hidden // n_heads // 2, 2, n_hidden).transpose(1, 2).reshape(n_hidden, n_hidden)

    print(f"Fetching all parameters from the checkpoint at {input_base_path}.")
    # Load weights
    if num_shards==1:
        # Not sharded
        # (The sharded implementation would also work, but this is simpler.)
        # /pure-mlo-scratch/alhernan/megatron-data/checkpoints/llama2-7b-tp4-pp1-optim/release/mp_rank_00/model_optim_rng.pt
        if os.path.exists(os.path.join(input_base_path, 'release')):
            filename = os.path.join(input_base_path, 'release', 'mp_rank_00', 'model_optim_rng.pt')
        elif input_base_path.split('/')[-1].startswith('iter_'):
            iteration = eval(input_base_path.split('/')[-1].replace('iter_', '').lstrip('0'))
            load_dir = '/'.join(input_base_path.split('/')[:-1])
            filename = os.path.join(input_base_path, 'mp_rank_00', 'model_optim_rng.pt')
            if not os.path.exists(filename):
                filename = filename.replace('model_optim_rng.pt', 'model_rng.pt')
        else:
            tracker_filename = os.path.join(input_base_path, 'latest_checkpointed_iteration.txt')
            with open(tracker_filename, 'r') as f:
                metastring = f.read().strip()
            iteration = 'iter_{:07d}'.format(int(metastring))
            filename = os.path.join(input_base_path, iteration, 'mp_rank_00', 'model_optim_rng.pt')
        if not os.path.exists(filename):
            filename = filename.replace('model_optim_rng.pt', 'model_rng.pt')
        original_filename = filename
        loaded = torch.load(filename, map_location="cpu")['model']['language_model']

    else:
        # Sharded
        filenames = []
        for i in range(num_shards):
            if os.path.exists(os.path.join(input_base_path, 'release')):
                filename = os.path.join(input_base_path, 'release', f'mp_rank_{i:02d}', 'model_optim_rng.pt')
            else:
                tracker_filename = os.path.join(input_base_path, 'latest_checkpointed_iteration.txt')
                with open(tracker_filename, 'r') as f:
                    metastring = f.read().strip()
                iteration = 'iter_{:07d}'.format(int(metastring))
                filename = os.path.join(input_base_path, iteration, f'mp_rank_{i:02d}', 'model_optim_rng.pt')
            if not os.path.exists(filename):
                filename = filename.replace('model_optim_rng.pt', 'model_rng.pt')
            filenames.append(filename)
        loaded = [
            torch.load(filenames[i], map_location="cpu")['model']['language_model']
            for i in range(num_shards)
        ]

    print('Llama-Megatron Loaded!')
    param_count = 0
    index_dict = {"weight_map": {}}
    
    print(f'Weighted Converting for {n_layers} layers...')
    for layer_i in range(n_layers):
        print(layer_i)
        filename = f"pytorch_model-{layer_i + 1}-of-{n_layers + 1}.bin"
        if num_shards == 1:
            # Unsharded
            state_dict = {
                f"model.layers.{layer_i}.self_attn.q_proj.weight": loaded['encoder'][f"layers.{layer_i}.self_attention.q_proj.weight"],
                f"model.layers.{layer_i}.self_attn.k_proj.multiway.0.weight": loaded['encoder'][f"layers.{layer_i}.self_attention.k_proj.multiway.0.weight"],
                f"model.layers.{layer_i}.self_attn.v_proj.multiway.0.weight": loaded['encoder'][f"layers.{layer_i}.self_attention.v_proj.multiway.0.weight"],
                f"model.layers.{layer_i}.self_attn.k_proj.multiway.1.weight": loaded['encoder'][f"layers.{layer_i}.self_attention.k_proj.multiway.1.weight"],
                f"model.layers.{layer_i}.self_attn.v_proj.multiway.1.weight": loaded['encoder'][f"layers.{layer_i}.self_attention.v_proj.multiway.1.weight"],
                f"model.layers.{layer_i}.self_attn.o_proj.weight": loaded['encoder'][f"layers.{layer_i}.self_attention.o_proj.weight"],
                f"model.layers.{layer_i}.mlp.gate_proj.weight": loaded['encoder'][f"layers.{layer_i}.mlp.gate_proj.weight"],
                f"model.layers.{layer_i}.mlp.down_proj.weight": loaded['encoder'][f"layers.{layer_i}.mlp.down_proj.weight"],
                f"model.layers.{layer_i}.mlp.up_proj.weight": loaded['encoder'][f"layers.{layer_i}.mlp.up_proj.weight"],
                f"model.layers.{layer_i}.input_layernorm.multiway.0.weight": loaded['encoder'][f"layers.{layer_i}.input_layernorm.multiway.0.weight"],
                f"model.layers.{layer_i}.post_attention_layernorm.multiway.0.weight": loaded['encoder'][f"layers.{layer_i}.post_attention_layernorm.multiway.0.weight"],
                f"model.layers.{layer_i}.input_layernorm.multiway.1.weight": loaded['encoder'][f"layers.{layer_i}.input_layernorm.multiway.1.weight"],
                f"model.layers.{layer_i}.post_attention_layernorm.multiway.1.weight": loaded['encoder'][f"layers.{layer_i}.post_attention_layernorm.multiway.1.weight"],
            }
        else:
            raise NotImplemented
#         else:
#             # Sharded
#             # Note that attention.w{q,k,v,o}, feed_fordward.w[1,2,3], attention_norm.weight and ffn_norm.weight share
#             # the same storage object, saving attention_norm and ffn_norm will save other weights too, which is
#             # redundant as other weights will be stitched from multiple shards. To avoid that, they are cloned.

#             state_dict = {
#                 f"model.layers.{layer_i}.input_layernorm.weight": loaded[0]['encoder'][
#                     f"layers.{layer_i}.input_layernorm.multiway.0.weight"
#                     ].clone(),
#                 f"model.layers.{layer_i}.post_attention_layernorm.weight": loaded[0]['encoder'][
#                     f"layers.{layer_i}.post_attention_layernorm.multiway.0.weight"
#                     ].clone(),
#             }
            
#             wqs, wks, wvs, ffn_w1s, ffn_w3s = [], [], [], [], []
#             for shard_idx in range(num_shards):
#                 wqs.append(loaded[shard_idx]['encoder'][f"layers.{layer_i}.self_attention.q_proj.weight"])
#                 wks.append(loaded[shard_idx]['encoder'][f"layers.{layer_i}.self_attention.k_proj.multiway.0.weight"])
#                 wvs.append(loaded[shard_idx]['encoder'][f"layers.{layer_i}.self_attention.v_proj.multiway.0.weight"])
                
#             state_dict[f"model.layers.{layer_i}.self_attn.q_proj.weight"] = permute(
#                 torch.cat(
#                     [
#                         wq.view(n_heads_per_shard, hidden_per_head, n_hidden)
#                         for wq in range(wqs)
#                     ],
#                     dim=0,
#                 ).reshape(n_hidden, n_hidden)
#             )
#             state_dict[f"model.layers.{layer_i}.self_attn.k_proj.weight"] = permute(
#                 torch.cat(
#                     [
#                         wk.view(n_heads_per_shard, hidden_per_head, n_hidden)
#                         for wk in range(wks)
#                     ],
#                     dim=0,
#                 ).reshape(n_hidden, n_hidden)
#             )
#             state_dict[f"model.layers.{layer_i}.self_attn.v_proj.weight"] = torch.cat(
#                 [
#                     wv.view(n_heads_per_shard, hidden_per_head, n_hidden)
#                         for wv in range(wvs)
#                 ],
#                 dim=0,
#             ).reshape(n_hidden, n_hidden)

#             state_dict[f"model.layers.{layer_i}.self_attn.o_proj.weight"] = torch.cat(
#                 [loaded[i]['encoder'][f"layers.{layer_i}.self_attention.o_proj.weight"] for i in range(num_shards)], dim=1
#             )
#             state_dict[f"model.layers.{layer_i}.mlp.gate_proj.weight"] = torch.cat(
#                 [loaded[i]['encoder'][f"layers.{layer_i}.mlp.gate_proj.weight"] for i in range(num_shards)], dim=0
#             )
#             state_dict[f"model.layers.{layer_i}.mlp.down_proj.weight"] = torch.cat(
#                 [loaded[i]['encoder'][f"layers.{layer_i}.mlp.down_proj.weight"] for i in range(num_shards)], dim=1
#             )
#             state_dict[f"model.layers.{layer_i}.mlp.up_proj.weight"] = torch.cat(
#                 [loaded[i]['encoder'][f"layers.{layer_i}.mlp.up_proj.weight"] for i in range(num_shards)], dim=0
#             )

        state_dict[f"model.layers.{layer_i}.self_attn.rotary_emb.inv_freq"] = inv_freq
        for k, v in state_dict.items():
            index_dict["weight_map"][k] = filename
            param_count += v.numel()
        torch.save(state_dict, os.path.join(tmp_model_path, filename))
        print(f'Sharded file saved to {filename}')

    filename = f"pytorch_model-{n_layers + 1}-of-{n_layers + 1}.bin"
    if num_shards==1:
        # Unsharded
        state_dict = {
            "model.embed_tokens.weight": loaded['embedding']['word_embeddings']['weight'],
            "model.norm.weight": loaded['encoder']['norm.weight'],
            "lm_head.weight": loaded['encoder']['lm_head.weight'],
        }
    else:
        state_dict = {
            "model.embed_tokens.weight": loaded[0]['embedding']['word_embeddings']['weight'],
            "model.norm.weight": loaded[0]['encoder']['norm.weight'],
            "lm_head.weight": loaded[0]['encoder']['lm_head.weight'],
        }
        
    
    loaded_all = torch.load(original_filename, map_location="cpu")['model']
    # Vision Part
    state_dict.update({
        "model.vision_model.embeddings.cls_token": loaded_all['vision_model']['cls_token'],
        "model.vision_model.embeddings.patch_embed.weight": loaded_all['vision_model']['patch_embed']['weight'],
        "model.vision_model.embeddings.position_embedding": loaded_all['vision_model']['position_embeddings'],
        "model.vision_model.embeddings.pre_layernorm.bias": loaded_all['vision_model']['pre_layernorm']['bias'],
        "model.vision_model.embeddings.pre_layernorm.weight": loaded_all['vision_model']['pre_layernorm']['weight'],
        "model.vision_model.post_layernorm.bias": loaded_all['vision_model']['transformer']['final_layernorm.bias'],
        "model.vision_model.post_layernorm.weight": loaded_all['vision_model']['transformer']['final_layernorm.weight'],
    })
    for v_layer_idx in range(24):
        state_dict.update({
            f"model.vision_model.encoder.layers.{v_layer_idx}.input_layernorm.bias": loaded_all['vision_model']['transformer'][f'layers.{v_layer_idx}.input_layernorm.bias'],
            f"model.vision_model.encoder.layers.{v_layer_idx}.input_layernorm.weight": loaded_all['vision_model']['transformer'][f'layers.{v_layer_idx}.input_layernorm.weight'],
            f"model.vision_model.encoder.layers.{v_layer_idx}.mlp.fc1.bias": loaded_all['vision_model']['transformer'][f'layers.{v_layer_idx}.mlp.dense_h_to_4h.bias'],
            f"model.vision_model.encoder.layers.{v_layer_idx}.mlp.fc1.weight": loaded_all['vision_model']['transformer'][f'layers.{v_layer_idx}.mlp.dense_h_to_4h.weight'],
            f"model.vision_model.encoder.layers.{v_layer_idx}.mlp.fc2.bias": loaded_all['vision_model']['transformer'][f'layers.{v_layer_idx}.mlp.dense_4h_to_h.bias'],
            f"model.vision_model.encoder.layers.{v_layer_idx}.mlp.fc2.weight": loaded_all['vision_model']['transformer'][f'layers.{v_layer_idx}.mlp.dense_4h_to_h.weight'],
            f"model.vision_model.encoder.layers.{v_layer_idx}.post_attention_layernorm.bias": loaded_all['vision_model']['transformer'][f'layers.{v_layer_idx}.post_attention_layernorm.bias'],
            f"model.vision_model.encoder.layers.{v_layer_idx}.post_attention_layernorm.weight": loaded_all['vision_model']['transformer'][f'layers.{v_layer_idx}.post_attention_layernorm.weight'],
            f"model.vision_model.encoder.layers.{v_layer_idx}.self_attn.dense.bias": loaded_all['vision_model']['transformer'][f'layers.{v_layer_idx}.self_attention.dense.bias'],
            f"model.vision_model.encoder.layers.{v_layer_idx}.self_attn.dense.weight": loaded_all['vision_model']['transformer'][f'layers.{v_layer_idx}.self_attention.dense.weight'],
            f"model.vision_model.encoder.layers.{v_layer_idx}.self_attn.query_key_value.bias": loaded_all['vision_model']['transformer'][f'layers.{v_layer_idx}.self_attention.query_key_value.bias'],
            f"model.vision_model.encoder.layers.{v_layer_idx}.self_attn.query_key_value.weight": loaded_all['vision_model']['transformer'][f'layers.{v_layer_idx}.self_attention.query_key_value.weight'],
        })

    # Abstractor Part
    state_dict.update({
        "model.visual_abstractor.query_embeds": loaded_all['vision_abstractor']['learnable_queries'],
        "model.visual_abstractor.visual_fc.bias": loaded_all['vision_abstractor']['visual_fc']['bias'],
        "model.visual_abstractor.visual_fc.weight": loaded_all['vision_abstractor']['visual_fc']['weight'],
        "model.visual_abstractor.vit_eos": loaded_all['vision_abstractor']['vit_eos'],
    })
    for v_layer_idx in range(6):
        state_dict.update({
            # f"model.visual_abstractor.encoder.layers.{v_layer_idx}.crossattention.attention.k_pos_embed": 
            f"model.visual_abstractor.encoder.layers.{v_layer_idx}.crossattention.attention.key.bias": loaded_all['vision_abstractor']['transformer'][f"layers.{v_layer_idx}.self_attention.k_proj.bias"],
            f"model.visual_abstractor.encoder.layers.{v_layer_idx}.crossattention.attention.key.weight": loaded_all['vision_abstractor']['transformer'][f"layers.{v_layer_idx}.self_attention.k_proj.weight"],
            # f"model.visual_abstractor.encoder.layers.{v_layer_idx}.crossattention.attention.q_pos_embed": "pytorch_model-00004-of-00004.bin",
            f"model.visual_abstractor.encoder.layers.{v_layer_idx}.crossattention.attention.query.bias": loaded_all['vision_abstractor']['transformer'][f"layers.{v_layer_idx}.self_attention.q_proj.bias"],
            f"model.visual_abstractor.encoder.layers.{v_layer_idx}.crossattention.attention.query.weight": loaded_all['vision_abstractor']['transformer'][f"layers.{v_layer_idx}.self_attention.q_proj.weight"],
            f"model.visual_abstractor.encoder.layers.{v_layer_idx}.crossattention.attention.value.bias": loaded_all['vision_abstractor']['transformer'][f"layers.{v_layer_idx}.self_attention.v_proj.bias"],
            f"model.visual_abstractor.encoder.layers.{v_layer_idx}.crossattention.attention.value.weight": loaded_all['vision_abstractor']['transformer'][f"layers.{v_layer_idx}.self_attention.v_proj.weight"],
            f"model.visual_abstractor.encoder.layers.{v_layer_idx}.crossattention.norm1.bias": loaded_all['vision_abstractor']['transformer'][f"layers.{v_layer_idx}.norm1.bias"],
            f"model.visual_abstractor.encoder.layers.{v_layer_idx}.crossattention.norm1.weight": loaded_all['vision_abstractor']['transformer'][f"layers.{v_layer_idx}.norm1.weight"],
            f"model.visual_abstractor.encoder.layers.{v_layer_idx}.crossattention.normk.bias": loaded_all['vision_abstractor']['transformer'][f"layers.{v_layer_idx}.normk.bias"],
            f"model.visual_abstractor.encoder.layers.{v_layer_idx}.crossattention.normk.weight": loaded_all['vision_abstractor']['transformer'][f"layers.{v_layer_idx}.normk.weight"],
            f"model.visual_abstractor.encoder.layers.{v_layer_idx}.crossattention.output.mlp.ffn_ln.bias": loaded_all['vision_abstractor']['transformer'][f"layers.{v_layer_idx}.mlp.ffn_ln.bias"],
            f"model.visual_abstractor.encoder.layers.{v_layer_idx}.crossattention.output.mlp.ffn_ln.weight": loaded_all['vision_abstractor']['transformer'][f"layers.{v_layer_idx}.mlp.ffn_ln.weight"],
            
            f"model.visual_abstractor.encoder.layers.{v_layer_idx}.crossattention.output.mlp.w1.bias": loaded_all['vision_abstractor']['transformer'][f"layers.{v_layer_idx}.mlp.w1.bias"],
            f"model.visual_abstractor.encoder.layers.{v_layer_idx}.crossattention.output.mlp.w1.weight": loaded_all['vision_abstractor']['transformer'][f"layers.{v_layer_idx}.mlp.w1.weight"],
            f"model.visual_abstractor.encoder.layers.{v_layer_idx}.crossattention.output.mlp.w2.bias": loaded_all['vision_abstractor']['transformer'][f"layers.{v_layer_idx}.mlp.w2.bias"],
            f"model.visual_abstractor.encoder.layers.{v_layer_idx}.crossattention.output.mlp.w2.weight": loaded_all['vision_abstractor']['transformer'][f"layers.{v_layer_idx}.mlp.w2.weight"],
            f"model.visual_abstractor.encoder.layers.{v_layer_idx}.crossattention.output.mlp.w3.bias": loaded_all['vision_abstractor']['transformer'][f"layers.{v_layer_idx}.mlp.w3.bias"],
            f"model.visual_abstractor.encoder.layers.{v_layer_idx}.crossattention.output.mlp.w3.weight": loaded_all['vision_abstractor']['transformer'][f"layers.{v_layer_idx}.mlp.w3.weight"],
            
            f"model.visual_abstractor.encoder.layers.{v_layer_idx}.crossattention.output.norm2.bias": loaded_all['vision_abstractor']['transformer'][f"layers.{v_layer_idx}.norm2.bias"],
            f"model.visual_abstractor.encoder.layers.{v_layer_idx}.crossattention.output.norm2.weight": loaded_all['vision_abstractor']['transformer'][f"layers.{v_layer_idx}.norm2.weight"],
            f"model.visual_abstractor.encoder.layers.{v_layer_idx}.crossattention.output.out_proj.bias": loaded_all['vision_abstractor']['transformer'][f"layers.{v_layer_idx}.self_attention.o_proj.bias"],
            f"model.visual_abstractor.encoder.layers.{v_layer_idx}.crossattention.output.out_proj.weight": loaded_all['vision_abstractor']['transformer'][f"layers.{v_layer_idx}.self_attention.o_proj.weight"],
        })

    for k, v in state_dict.items():
        index_dict["weight_map"][k] = filename
        param_count += v.numel()
    torch.save(state_dict, os.path.join(tmp_model_path, filename))

    # Write configs
    index_dict["metadata"] = {"total_size": param_count * 2}
    write_json(index_dict, os.path.join(tmp_model_path, "pytorch_model.bin.index.json"))

    config = MPLUGOwl2Config()
    config.save_pretrained(tmp_model_path)

    # Make space so we can load the model properly now.
    del state_dict
    del loaded
    del loaded_all
    gc.collect()

def write_tokenizer(tokenizer_path, input_tokenizer_path):
    # Initialize the tokenizer based on the `spm` model
    tokenizer_class = LlamaTokenizer if LlamaTokenizerFast is None else LlamaTokenizerFast
    print(f"Saving a {tokenizer_class.__name__} to {tokenizer_path}.")
    tokenizer = tokenizer_class(input_tokenizer_path)
    tokenizer.save_pretrained(tokenizer_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        help="Location of LLaMA_Megatron weights",
    )
    parser.add_argument(
        "--model_size",
        type=int,
        default=7,
        choices=[7, 13, 30, 65, 70],
    )
    parser.add_argument(
        "--num_input_shards",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--num_output_shards",
        type=int,
        default=1,
    )
    parser.add_argument('--skip_permute', action='store_true')
    
    parser.add_argument(
        "--output_dir",
        help="Location to write HF model and tokenizer",
    )
    
    args = parser.parse_args()
    write_model(
        model_path=args.output_dir,
        input_base_path=args.input_dir,
        model_size=args.model_size,
        num_input_shards=args.num_input_shards,
        num_output_shards=args.num_output_shards,
        skip_permute=args.skip_permute
    )
    

if __name__ == "__main__":
    main()
