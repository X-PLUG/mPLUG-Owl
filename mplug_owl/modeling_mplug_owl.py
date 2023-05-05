# coding=utf-8
# Copyright 2023 The Salesforce Authors and The HuggingFace Team. All rights reserved.
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

import base64
import datetime
import json
import math
import os
import re
import sys
import threading
from dataclasses import dataclass
from io import BytesIO
from typing import Any, List, Optional, Tuple, Union

import requests
import torch
import torch.utils.checkpoint
from apex.normalization import MixedFusedLayerNorm
from PIL import Image
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torchvision import transforms
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPooling, BaseModelOutputWithPoolingAndCrossAttentions)
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto import (AutoModelForCausalLM,
                                      AutoModelForSeq2SeqLM)
from transformers.pytorch_utils import (apply_chunking_to_forward,
                                        find_pruneable_heads_and_indices,
                                        prune_linear_layer)
from transformers.utils import (ModelOutput, add_start_docstrings,
                                add_start_docstrings_to_model_forward, logging,
                                replace_return_docstrings)

from clip.modeling_clip import CLIPVisionTransformer
from icecream import ic
from .configuration_mplug_owl import (mPLUG_OwlConfig,
                                      mPLUG_OwlVisualAbstractorConfig)

logger = logging.get_logger(__name__)


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"


lock = threading.Lock()


logger = logging.get_logger(__name__)


def base64decode(s: str):
    """
    Decode base64 `str` to original `bytes`.
    If the input is not a valid base64 string, return None.

    Args:
        s(str): A base64 `str` that can be used in text file.

    Returns:
        Optional[bytes]: The original decoded data with type `bytes`.
            If the input is not a valid base64 string, return None.
    """
    # return base64.b64decode(s)
    _base64_regex = re.compile(
        r'^(?:[A-Za-z\d+/]{4})*(?:[A-Za-z\d+/]{3}=|[A-Za-z\d+/]{2}==)?$')
    s = s.translate(base64._urlsafe_decode_translation)
    if not _base64_regex.fullmatch(s):
        return None
    try:
        return base64.urlsafe_b64decode(s)
    except base64.binascii.Error:
        return None


@dataclass
class mPLUG_OwlForConditionalGenerationModelOutput(ModelOutput):
    """
    Class defining the outputs of [`mPLUG_OwlForConditionalGeneration`].

    Args:
        loss (`torch.FloatTensor`, *optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Language modeling loss from the language model.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head of the language model.
        vision_outputs (`BaseModelOutputWithPooling`):
            Outputs of the vision encoder.
      
        language_model_outputs (`CausalLMOutputWithPast` or `Seq2SeqLMOutput`):
            Outputs of the language model.
    """

    loss: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    vision_outputs: Optional[torch.FloatTensor] = None
    language_model_outputs: Optional[Tuple[torch.FloatTensor]] = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k]
            if k not in ["vision_outputs", "language_model_outputs"]
            else getattr(self, k).to_tuple()
            for k in self.keys()
        )


class mPLUG_OwlPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = mPLUG_OwlConfig
    base_model_prefix = "mplug_owl"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [
        r"position_ids",
        r"language_model.encoder.embed_tokens.weight",
        r"language_model.decoder.embed_tokens.weight",
        r"language_model.lm_head.weight",
    ]
    _no_split_modules = ["mPLUG_OwlAttention", "T5Block", "OPTDecoderLayer"]
    _keep_in_fp32_modules = ["wo"]

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_range
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Embedding) or isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=factor)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, CLIPVisionTransformer):
            module.gradient_checkpointing = value


class mPLUG_OwlVisualAbstractorMultiHeadAttention(nn.Module):
    def __init__(self, config, is_cross_attention=False, add_bias_kv=True):
        super().__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention heads (%d)"
                % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        if is_cross_attention:
            self.key = nn.Linear(
                config.encoder_hidden_size, self.all_head_size)
            self.value = nn.Linear(
                config.encoder_hidden_size, self.all_head_size)
        else:
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(
                2 * config.max_position_embeddings - 1, self.attention_head_size)
        self.save_attention = False

        if add_bias_kv:
            from torch.nn import Parameter
            self.bias_k = Parameter(torch.empty((1, 1, config.hidden_size)))
            self.bias_v = Parameter(torch.empty((1, 1, config.hidden_size)))
        else:
            self.bias_k = self.bias_v = None

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[
            :-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def _pad_masks(
        self,
        key_padding_mask,
    ):
        shape = key_padding_mask.size()[:-1] + torch.Size([1])
        key_padding_mask = torch.cat(
            [
                key_padding_mask,
                key_padding_mask.new_zeros(shape),
            ],
            dim=-1,
        )
        return key_padding_mask

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention:
            key_layer = self.key(encoder_hidden_states)
            value_layer = self.value(encoder_hidden_states)
            if self.bias_k is not None:
                key_layer = torch.cat([key_layer, self.bias_k.repeat(
                    hidden_states.shape[0], 1, 1)], dim=1)  # B L D
                value_layer = torch.cat(
                    [value_layer, self.bias_v.repeat(hidden_states.shape[0], 1, 1)], dim=1)
                encoder_attention_mask = self._pad_masks(
                    encoder_attention_mask)
            key_layer = self.transpose_for_scores(key_layer)
            value_layer = self.transpose_for_scores(value_layer)
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        mixed_query_layer = self.query(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)

        past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(
                seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(
                seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(
                distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(
                dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum(
                    "bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + \
                    relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / \
            math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        if is_cross_attention and self.save_attention:
            self.save_attention_map(attention_probs)
            attention_probs.register_hook(self.save_attn_gradients)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs_dropped = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs_dropped = attention_probs_dropped * head_mask

        context_layer = torch.matmul(attention_probs_dropped, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
            :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (
            context_layer,)

        outputs = outputs + (past_key_value,)
        return outputs


class SwiGU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        multiple_of = 256
        hidden_features = multiple_of * \
            ((hidden_features + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.w2 = nn.Linear(hidden_features, out_features)
        self.w3 = nn.Linear(in_features, hidden_features)
        self.ffn_ln = MixedFusedLayerNorm(hidden_features, eps=1e-6)

    def forward(self, x):
        return self.w2(self.ffn_ln(self.act(self.w1(x)) * self.w3(x)))


class mPLUG_OwlVisualAbstractorSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        dim = config.hidden_size
        self.out_proj = nn.Linear(dim, dim, bias=True)
        self.norm2 = MixedFusedLayerNorm(dim)
        self.mlp = SwiGU(in_features=dim, hidden_features=4 *
                         dim, act_layer=nn.SiLU)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        input_tensor = input_tensor + self.out_proj(hidden_states)
        input_tensor = input_tensor + self.mlp(self.norm2(input_tensor))
        return input_tensor


class mPLUG_OwlVisualAbstractorAttention(nn.Module):
    def __init__(self, config, is_cross_attention=False):
        super().__init__()
        self.attention = mPLUG_OwlVisualAbstractorMultiHeadAttention(
            config, is_cross_attention)
        self.output = mPLUG_OwlVisualAbstractorSelfOutput(config)
        self.pruned_heads = set()
        self.norm1 = MixedFusedLayerNorm(config.hidden_size)
        self.normk = MixedFusedLayerNorm(config.hidden_size)

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(
            self.output.out_proj, index, dim=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - \
            len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * \
            self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # HACK we apply norm on q and k
        hidden_states = self.norm1(hidden_states)
        encoder_hidden_states = self.normk(encoder_hidden_states)
        encoder_hidden_states = torch.cat(
            [hidden_states, encoder_hidden_states], dim=1)
        encoder_attention_mask = torch.cat(
            [attention_mask, encoder_attention_mask], dim=-1)
        self_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        # add attentions if we output them
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class mPLUG_OwlVisualAbstractorIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class mPLUG_OwlVisualAbstractorOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = MixedFusedLayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class mPLUG_OwlVisualAbstractorLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1

        self.layer_idx = layer_idx

        self.crossattention = mPLUG_OwlVisualAbstractorAttention(
            config, is_cross_attention=True)
        self.has_cross_attention = True

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        query_length=0,
    ):
        # HACK we do not perform self attention on query
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        # self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # self_attention_outputs = self.attention(
        #     hidden_states,
        #     attention_mask,
        #     head_mask,
        #     output_attentions=output_attentions,
        #     past_key_value=self_attn_past_key_value,
        # )

        # attention_output = self_attention_outputs[0]
        # outputs = self_attention_outputs[1:-1]

        # present_key_value = self_attention_outputs[-1]
        attention_output = hidden_states
        query_attention_output = attention_output[:, :query_length, :]

        if self.has_cross_attention:
            if encoder_hidden_states is None:
                raise ValueError(
                    "encoder_hidden_states must be given for cross-attention layers")
            cross_attention_outputs = self.crossattention(
                query_attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions=output_attentions,
            )
            query_attention_output = cross_attention_outputs[0]

        outputs = (query_attention_output,)
        return outputs


class mPLUG_OwlVisualAbstractorEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [mPLUG_OwlVisualAbstractorLayer(
                config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        query_length=0,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions else None

        next_decoder_cache = () if use_cache else None

        for i in range(self.config.num_hidden_layers):
            layer_module = self.layers[i]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:
                if use_cache:
                    logger.warn(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions, query_length)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    query_length,
                )

            hidden_states = layer_outputs[0]
            # if use_cache:
            #     next_decoder_cache += (layer_outputs[-1],)
            # if output_attentions:
            #     all_self_attentions = all_self_attentions + (layer_outputs[1],)
            #     if layer_module.has_cross_attention:
            #         all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # if output_hidden_states:
        #     all_hidden_states = all_hidden_states + (hidden_states,)

        # if not return_dict:
        #     return tuple(
        #         v
        #         for v in [
        #             hidden_states,
        #             next_decoder_cache,
        #             all_hidden_states,
        #             all_self_attentions,
        #             all_cross_attentions,
        #         ]
        #         if v is not None
        #     )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            # past_key_values=next_decoder_cache,
            # hidden_states=all_hidden_states,
            # attentions=all_self_attentions,
            # cross_attentions=all_cross_attentions,
        )


class mPLUG_OwlVisualAbstractorModel(mPLUG_OwlPreTrainedModel):

    def __init__(self, config: mPLUG_OwlVisualAbstractorConfig, language_hidden_size):
        super().__init__(config)
        self.config = config

        # self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.encoder = mPLUG_OwlVisualAbstractorEncoder(config)
        self.visual_fc = torch.nn.Linear(
            config.hidden_size, language_hidden_size)
        self.vit_eos = torch.nn.Parameter(
            torch.randn(1, 1, language_hidden_size)
        )
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def get_extended_attention_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: Tuple[int],
        device: torch.device,
        has_query: bool = False,
    ) -> torch.Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.
            device: (`torch.device`):
                The device of the input to the model.

        Returns:
            `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - the model is an encoder, so make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(
        self,
        query_embeds,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of:
            shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`): Contains precomputed key and
            value hidden states of the attention blocks. Can be used to speed up decoding. If `past_key_values` are
            used, the user can optionally input only the last `decoder_input_ids` (those that don't have their past key
            value states given to this model) of shape `(batch_size, 1)` instead of all `decoder_input_ids` of shape
            `(batch_size, sequence_length)`.
        use_cache (`bool`, `optional`):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] -
            self.config.query_length if past_key_values is not None else 0
        )

        query_length = query_embeds.shape[1] if query_embeds is not None else 0
        # HACK we do not use layernorm and dropout
        # embedding_output = self.layernorm(query_embeds)
        # embedding_output = self.dropout(embedding_output)
        embedding_output = query_embeds
        input_shape = embedding_output.size()[:-1]
        batch_size, seq_length = input_shape
        device = embedding_output.device

        if attention_mask is None:
            attention_mask = torch.ones(
                ((batch_size, seq_length + past_key_values_length)), device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if encoder_hidden_states is not None:
            if type(encoder_hidden_states) == list:
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states[0].size(
                )
            else:
                (
                    encoder_batch_size,
                    encoder_sequence_length,
                    _,
                ) = encoder_hidden_states.size()
            encoder_hidden_shape = (
                encoder_batch_size, encoder_sequence_length)

            if type(encoder_attention_mask) == list:
                encoder_extended_attention_mask = [
                    self.invert_attention_mask(mask) for mask in encoder_attention_mask]
            elif encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=device)
                encoder_extended_attention_mask = self.invert_attention_mask(
                    encoder_attention_mask)
            else:
                encoder_extended_attention_mask = self.invert_attention_mask(
                    encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(
            head_mask, self.config.num_hidden_layers)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            query_length=query_length,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = sequence_output[:, 0, :]

        sequence_output = self.visual_fc(sequence_output)
        sequence_output = torch.cat(
            [sequence_output, self.vit_eos.repeat(sequence_output.shape[0], 1, 1)], dim=1)

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            # past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            # attentions=encoder_outputs.attentions,
            # cross_attentions=encoder_outputs.cross_attentions,
        )


class ImageProcessor(object):
    def __init__(self, resolution=224):
        normalize = transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution),
                              interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ])

    def __call__(self, image_paths):
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        images = []
        for image_path in image_paths:
            if image_path.startswith("http://") or image_path.startswith("https://"):
                # We need to actually check for a real protocol, otherwise it's impossible to
                # use a local file like http_huggingface_co.png.
                image = Image.open(requests.get(image_path, stream=True).raw)
            elif os.path.exists(image_path):
                image = Image.open(image_path).convert('RGB')
            else:
                image_bytes = base64decode(image_path)
                if image_bytes is not None:
                    image = Image.open(BytesIO(image_bytes)).convert('RGB')
                elif os.path.isfile(image_path):
                    image = Image.open(image_path).convert('RGB')

            image = self.transform(image).unsqueeze(0)
            images.append(image)
        images = torch.cat(images, dim=0)
        return images


def get_ltor_masks_and_position_ids_from_embeddings(data):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    micro_batch_size, seq_length = data.size()[:2]

    # Attention mask (lower triangular).
    att_mask_batch = 1
    attention_mask = torch.tril(torch.ones(
        (att_mask_batch, seq_length, seq_length), device=data.device)).view(
            att_mask_batch, 1, seq_length, seq_length)

    # Loss mask.
    loss_mask = torch.ones(
        data.size()[:2], dtype=torch.float, device=data.device)

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data[..., 0])

    # Convert attention mask to binary:
    attention_mask = (attention_mask < 0.5)

    return attention_mask, loss_mask, position_ids


class mPLUG_OwlModel(mPLUG_OwlPreTrainedModel):
    config_class = mPLUG_OwlConfig
    main_input_name = "pixel_values"

    def __init__(self, config: mPLUG_OwlConfig):
        super().__init__(config)

        from clip.modeling_clip import CLIPVisionTransformer

        # we hack the source code in CLIPVisionTransformer.
        self.vision_model = CLIPVisionTransformer(config.vision_config)

        self.query_tokens = nn.Parameter(torch.zeros(
            1, config.num_query_tokens, config.visual_abstractor_config.hidden_size))
        self.abstractor = mPLUG_OwlVisualAbstractorModel(
            config.visual_abstractor_config, config.text_config.hidden_size)

        # if config.use_decoder_only_language_model:
        from llama.modeling_llama import LlamaForCausalLM
        language_model = LlamaForCausalLM(config=config.text_config)
        # language_model = AutoModelForCausalLM.from_pretrained('/nas-alinlp/butyuhao/llama-7b-hf')
        # else:
        #     language_model = AutoModelForSeq2SeqLM.from_config(config.text_config)
        self.language_model = language_model
        self.vit_eval = self.config.vit_eval if hasattr(self.config,'vit_eval') else False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def get_encoder(self):
        return self.language_model.get_encoder()

    def get_decoder(self):
        return self.language_model.get_decoder()

    def _tie_weights(self):
        if not self.config.use_decoder_only_language_model:
            self.language_model.encoder.embed_tokens = self.language_model.shared
            self.language_model.decoder.embed_tokens = self.language_model.shared

    
def get_media_indices(my_list):
    if isinstance(my_list, torch.Tensor):
        my_list = my_list.cpu().tolist()
    result = []
    for i in range(len(my_list)):
        if i == 0 and my_list[i] < 0:
            result.append(i)
        elif my_list[i] != my_list[i-1] and my_list[i] < 0:
            result.append(i)
    return result

class mPLUG_OwlForConditionalGeneration(mPLUG_OwlPreTrainedModel):
    config_class = mPLUG_OwlConfig
    main_input_name = "pixel_values"

    def __init__(self, config: mPLUG_OwlConfig):
        super().__init__(config)

        # we hack the source code in CLIPVisionTransformer.
        self.vision_model = CLIPVisionTransformer(config.vision_config)

        self.query_tokens = nn.Parameter(torch.zeros(
            1, config.num_query_tokens, config.visual_abstractor_config.hidden_size))
        self.abstractor = mPLUG_OwlVisualAbstractorModel(
            config.visual_abstractor_config, config.text_config.hidden_size)

        # if config.use_decoder_only_language_model:
        from llama.modeling_llama import LlamaForCausalLM
        language_model = LlamaForCausalLM(config=config.text_config)
        # language_model = AutoModelForCausalLM.from_pretrained('/nas-alinlp/butyuhao/llama-7b-hf')
        # else:
        #     language_model = AutoModelForSeq2SeqLM.from_config(config.text_config)
        self.language_model = language_model
        self.vit_eval = self.config.vit_eval if hasattr(self.config,'vit_eval') else False
        # Initialize weights and apply final processing
        self.post_init()
        self.main_input_name = 'input_ids'
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def get_encoder(self):
        return self.language_model.get_encoder()

    def get_decoder(self):
        return self.language_model.get_decoder()

    def _tie_weights(self):
        pass
        # if not self.config.use_decoder_only_language_model:
        #     self.language_model.encoder.embed_tokens = self.language_model.shared
        #     self.language_model.decoder.embed_tokens = self.language_model.shared

    def _preprocess_accelerate(self):
        r"""
        Some pre-processing hacks to make the model `accelerate` compatible. Check
        https://github.com/huggingface/transformers/pull/21707 for more details.
        """
        hf_device_map = self.hf_device_map

        if len(hf_device_map) > 1 and "language_model" not in hf_device_map and torch.cuda.device_count() > 1:
            logger.warning(
                "The `language_model` is not in the `hf_device_map` dictionary and you are running your script"
                " in a multi-GPU environment. this may lead to unexpected behavior when using `accelerate`."
                " Please pass a `device_map` that contains `language_model` to remove this warning."
                " Please refer to https://github.com/huggingface/blog/blob/main/accelerate-large-models.md for",
                " more details on creating a `device_map` for large models.",
            )

        if hasattr(self.language_model, "_hf_hook"):
            self.language_model._hf_hook.io_same_device = True  # For `generate` compatibility

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.FloatTensor,
        num_images,
        non_padding_mask: Optional[torch.LongTensor] = None,
        non_media_mask: Optional[torch.LongTensor] = None,
        prompt_mask: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, mPLUG_OwlForConditionalGenerationModelOutput]:
  
        # get text embedding
        text_tokens_ = input_ids
        batch_size = input_ids.shape[0]
        # labels = text_tokens_[:, 1:].clone().contiguous()

        media_token_indices = [
            get_media_indices(text_tokens_[i][:-1]) # [:-1] since we would not use the last token for embedding
            for i in range(batch_size)
        ]
        text_tokens_[text_tokens_ < 0] = 1 # Not used
        # text_tokens = text_tokens_[:, :-1].contiguous()
        text_embeds = self.get_input_embeddings()(text_tokens_) # Temporally Embedding
        
        if pixel_values is not None:
            if self.vit_eval:
                with torch.no_grad():
                    image_embeds = self.vision_model(pixel_values, return_dict=True).last_hidden_state
            else:
                image_embeds = self.vision_model(pixel_values, return_dict=True).last_hidden_state
            
           
            image_attention_mask = torch.ones(
                image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
            query_tokens = self.query_tokens.expand(
                image_embeds.shape[0], -1, -1)
            
            query_features = self.abstractor(query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,)['last_hidden_state']
            query_atts = torch.ones(query_features.size(
            )[:-1], dtype=torch.long).to(query_features.device)
            img_seq_length = query_features.shape[1]

        num_images_per_sample = num_images.long().cpu().tolist()
  

        text_chunk_embeds = []
        img_idx = 0
        for b in range(batch_size):
            start = 0
            result = []
            if len(media_token_indices[b]) > 0:
                for i, pos in enumerate(media_token_indices[b]):
                    if pos > start:
                        result.append(text_embeds[b, start:pos])
                    result.append(query_features[img_idx+i])
                    start = pos + img_seq_length
            if start < text_embeds.shape[1]:
                result.append(text_embeds[b, start:])

            img_idx += num_images_per_sample[b]
            text_chunk_embeds.append(torch.cat(result, dim=0))

        # Actual Input Embeddings
        input_embeds = torch.stack(text_chunk_embeds, dim=0)

        if pixel_values is None and self.language_model.is_gradient_checkpointing:
            # Hack here when gradient checkpoint is enable.
            # Keep the compute graph static
            image_embeds = self.vision_model(torch.zeros(1,3,224,224,device=input_embeds.device,dtype=input_embeds.dtype), return_dict=True).last_hidden_state
            query_tokens = self.query_tokens.expand(
                image_embeds.shape[0], -1, -1)
            query_features = self.abstractor(query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,)['last_hidden_state']

            input_embeds = input_embeds + query_features.mean()*0

        # Create causal mask and position ids
        _, loss_mask, position_ids = \
            get_ltor_masks_and_position_ids_from_embeddings(input_embeds)

        # Calculate the loss_mask
        non_padding_mask = non_padding_mask.long()
        non_media_mask = non_media_mask.long()
        prompt_mask = prompt_mask.long() # TODO How to deal with prompt mask
        # from icecream import ic
        # non_padding_mask = non_padding_mask[:,:-1]
        # non_media_mask = non_media_mask[:,:-1]
        # prompt_mask = prompt_mask[:,:-1]
        # attention_mask = attention_mask[:,:-1]
        loss_mask=loss_mask[:,:-1]

        loss_mask = loss_mask * non_padding_mask * non_media_mask * prompt_mask
        
        # Forward into GPT
        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        outputs.loss = (outputs.loss * loss_mask.view(-1)).sum()/loss_mask.sum()
        return outputs

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        """
        Overrides `generate` function to be able to use the model as a conditional generator.

        Args:
            pixel_values (`torch.FloatTensor` of shape (batch_size, num_channels, height, width)):
                Input images to be processed.
            input_ids (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                The sequence used as a prompt for the generation.
            attention_mask (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                Mask to avoid performing attention on padding token indices

        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """

        
        if input_ids is not None:
            batch_size = input_ids.size(0)
            media_token_indices = [
                get_media_indices(input_ids[i])
                for i in range(batch_size)
            ]
            num_images_per_sample = [len(x) for x in media_token_indices]
            input_ids[input_ids < 0] = 0  # Not used

        if hasattr(self, "hf_device_map"):
            # preprocess for `accelerate`
            self._preprocess_accelerate()
        print(input_ids.shape)
        batch_size = input_ids.shape[0]
        # get text embedding
        inputs_embeds = self.get_input_embeddings()(input_ids)
        # get visual embedding
        if pixel_values is not None:
            pixel_values = pixel_values.to(input_ids.device)
            with torch.no_grad():
                print(pixel_values.shape)
                image_embeds = self.vision_model(
                    pixel_values, return_dict=True).last_hidden_state
                image_attention_mask = torch.ones(
                    image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
                query_tokens = self.query_tokens.expand(
                    image_embeds.shape[0], -1, -1)
                query_outputs = self.abstractor(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_attention_mask,
                    return_dict=True,
                )
                query_output = query_outputs['last_hidden_state']
                image_embeds = query_output
            img_seq_length = image_embeds.shape[1]

            # ===================
            # Get actual input embeddings
            # ===================
            text_chunk_embeds = []
            text_chunk_attns = []
            img_idx = 0

            for b in range(batch_size):
                start = 0
                result = []
                result_attn = []
                for i, pos in enumerate(media_token_indices[b]):
                    if pos > start:
                        result.append(inputs_embeds[b, start:pos])
                        result_attn.append(attention_mask[b, start:pos])
                    result.append(image_embeds[img_idx+i])
                    result_attn.append(torch.ones(
                        image_embeds[img_idx+i].shape[0], device=inputs_embeds.device))
                    start = pos + img_seq_length
                if start < inputs_embeds.shape[1]:
                    result.append(inputs_embeds[b, start:])
                    result_attn.append(attention_mask[b, start:])

                img_idx += num_images_per_sample[b]
                text_chunk_embeds.append(torch.cat(result, dim=0))
                text_chunk_attns.append(torch.cat(result_attn, dim=0))
            inputs_embeds = torch.stack(text_chunk_embeds, dim=0)
            attention_mask = torch.stack(text_chunk_attns, dim=0)

        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            # input_ids=input_ids,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

        return outputs
