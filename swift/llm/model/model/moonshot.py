# Copyright (c) Alibaba, Inc. and its affiliates.
from swift.llm import TemplateType
from ..constant import LLMModelType, MLLMModelType
from ..model_arch import ModelArch
from ..register import (Model, ModelGroup, ModelMeta, get_model_tokenizer_multimodal,
                        get_model_tokenizer_with_flash_attn, register_model)

register_model(
    ModelMeta(
        LLMModelType.moonlight,
        [
            ModelGroup([
                Model('moonshotai/Moonlight-16B-A3B', 'moonshotai/Moonlight-16B-A3B'),
                Model('moonshotai/Moonlight-16B-A3B-Instruct', 'moonshotai/Moonlight-16B-A3B-Instruct'),
            ]),
        ],
        TemplateType.moonlight,
        get_model_tokenizer_with_flash_attn,
        architectures=['DeepseekV3ForCausalLM'],
        model_arch=ModelArch.deepseek_v2,
        requires=['transformers<4.49'],
    ))

register_model(
    ModelMeta(
        MLLMModelType.kimi_vl,
        [
            ModelGroup([
                Model('moonshotai/Kimi-VL-A3B-Instruct', 'moonshotai/Kimi-VL-A3B-Instruct'),
                Model('moonshotai/Kimi-VL-A3B-Thinking', 'moonshotai/Kimi-VL-A3B-Thinking'),
            ])
        ],
        TemplateType.kimi_vl,
        get_model_tokenizer_multimodal,
        architectures=['KimiVLForConditionalGeneration'],
        model_arch=ModelArch.llava_hf,
        requires=['transformers<4.49'],
    ))
