# coding=utf-8
# Copyright 2023 Alibaba Inc. and The HuggingFace Inc. team. All rights reserved.
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


import copy
import os
from typing import Union

from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import \
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.utils import logging

logger = logging.get_logger(__name__)


class mPLUG_OwlVisualAbstractorConfig(PretrainedConfig):

    model_type = "mPLUG_OwlVisualAbstractor"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=1024,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        pad_token_id=0,
        position_embedding_type="absolute",
        classifier_dropout=None,
        cross_attention_frequency=2,
        encoder_hidden_size=1024,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.classifier_dropout = classifier_dropout
        self.cross_attention_frequency = cross_attention_frequency
        self.encoder_hidden_size = encoder_hidden_size

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        config_dict, kwargs = cls.get_config_dict(
            pretrained_model_name_or_path, **kwargs)

        if config_dict.get("model_type") == "mplug_owl":
            config_dict = config_dict["abstractor_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class mPLUG_OwlConfig(PretrainedConfig):

    model_type = "mplug_owl"
    is_composition = True

    def __init__(self, vision_config=None, visual_abstractor_config=None, text_config=None, num_query_tokens=64, **kwargs):
        super().__init__(**kwargs)
        from clip.configuration_clip import CLIPVisionConfig
        if vision_config is None:
            # By defalt we use openai-clip large patch14

            vision_config = CLIPVisionConfig(
                **vision_config_dict, layer_norm_eps=1e-6).to_dict()
            logger.info(
                "vision_config is None.")

        if visual_abstractor_config is None:
            visual_abstractor_config = {}
            logger.info(
                "abstractor_config is None. ")

        if text_config is None:
            # we use LLAMA 7b by default
            from transformers.models.llama.configuration_llama import \
                LlamaConfig
            text_config = LlamaConfig(pad_token_id=2).to_dict()
            logger.info("text_config is None.")

        self.vision_config = CLIPVisionConfig(**vision_config)
        self.visual_abstractor_config = mPLUG_OwlVisualAbstractorConfig(
            **visual_abstractor_config)
        self.visual_abstractor_config.layer_norm_eps = 1e-6
        text_model_type = text_config["model_type"] if "model_type" in text_config else "opt"
        self.text_config = CONFIG_MAPPING[text_model_type](**text_config)

        self.tie_word_embeddings = self.text_config.tie_word_embeddings
        self.is_encoder_decoder = self.text_config.is_encoder_decoder

        self.num_query_tokens = num_query_tokens
        self.visual_abstractor_config.encoder_hidden_size = self.vision_config.hidden_size
        self.use_decoder_only_language_model = self.text_config.model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
        self.initializer_factor = 1.0
        self.initializer_range = 0.02

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["vision_config"] = self.vision_config.to_dict()
        output["abstractor_config"] = self.visual_abstractor_config.to_dict()
        output["text_config"] = self.text_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output


vision_config_dict = {
    "hidden_size": 1024,
    "intermediate_size": 4096,
    "num_attention_heads": 8,
    "num_hidden_layers": 24,
    "patch_size": 14,
    "projection_dim": 768}
