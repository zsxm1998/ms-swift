"""OmniPath model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers.models.auto import CONFIG_MAPPING


logger = logging.get_logger(__name__)


class OmniPathConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`OmniPathForConditionalGeneration`]. It is used to instantiate an
    OmniPath model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the LLaVA-9B.

    e.g. [llava-hf/llava-9b](https://huggingface.co/llava-hf/llava-9b)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (`Union[AutoConfig, dict]`,  *optional*, defaults to `CLIPVisionConfig`):
            The config object or dictionary of the vision backbone.
        text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `LlamaConfig`):
            The config object or dictionary of the text backbone.
        ignore_index (`int`, *optional*, defaults to -100):
            The ignore index for the loss function.
        image_token_index (`int`, *optional*, defaults to 32000):
            The image token index to encode the image prompt.
        mask_token_index (`int`, *optional*, defaults to 32001):
            The mask token index to encode the mask prompt.
        patch_token_index (`int`, *optional*, defaults to 32002):
            The patch token index to encode the patches provided by the function calling.
        projector_hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The activation function used by the multimodal projector.
        vision_feature_select_strategy (`str`, *optional*, defaults to `"default"`):
            The feature selection strategy used to select the vision feature from the vision backbone.
            Can be one of `"default"` or `"full"`.
        vision_feature_layer (`int`, *optional*, defaults to -2):
            The index of the layer to select the vision feature.
        image_seq_length (`int`, *optional*, defaults to 576):
            Sequence length of one image embedding.

    Example:

    ```python
    >>> from transformers import CLIPVisionConfig, LlamaConfig
    >>> from .modeling_omnipath import OmniPathForConditionalGeneration

    >>> # Initializing a CLIP-vision config
    >>> vision_config = CLIPVisionConfig()

    >>> # Initializing a Llama config
    >>> text_config = LlamaConfig()

    >>> # Initializing a llava-1.5-7b style configuration
    >>> configuration = OmniPathConfig(vision_config, text_config)

    >>> # Initializing a model from the llava-1.5-7b style configuration
    >>> model = OmniPathForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "omnipath"
    is_composition = True

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_index=-100,
        image_token_index=32000,
        mask_token_index=32001,
        patch_token_index=32002,
        projector_hidden_act="gelu",
        vision_feature_select_strategy="default",
        vision_feature_layer=-2,
        image_seq_length=576,
        **kwargs,
    ):
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.mask_token_index = mask_token_index
        self.patch_token_index = patch_token_index
        self.projector_hidden_act = projector_hidden_act
        self.image_seq_length = image_seq_length

        if vision_feature_select_strategy not in ["default", "full"]:
            raise ValueError(
                "vision_feature_select_strategy should be one of 'default', 'full'."
                f"Got: {vision_feature_select_strategy}"
            )

        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer

        if isinstance(vision_config, dict):
            vision_config["model_type"] = (
                vision_config["model_type"] if "model_type" in vision_config else "clip_vision_model"
            )
            vision_config = CONFIG_MAPPING[vision_config["model_type"]](**vision_config)
        elif vision_config is None:
            vision_config = CONFIG_MAPPING["clip_vision_model"](
                intermediate_size=4096,
                hidden_size=1024,
                patch_size=14,
                image_size=336,
                num_hidden_layers=24,
                num_attention_heads=16,
                vocab_size=32000,
                projection_dim=768,
            )

        self.vision_config = vision_config

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "llama"
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["llama"]()

        self.text_config = text_config

        super().__init__(**kwargs)
