from transformers.configuration_utils import PretrainedConfig
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLVisionConfig, Qwen2VLConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING


class OmniPTQwen2VLSecondVisionConfig(PretrainedConfig):
    model_type = "omnipt_qwen2_vl"
    base_config_key = "second_vision_config"

    def __init__(
        self,
        vit_type="timm",
        model_name="hf-hub:paige-ai/Virchow2",
        vision_feature_layer=-2,
        hidden_size=3584,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vit_type = vit_type
        self.model_name = model_name
        self.vision_feature_layer = vision_feature_layer if isinstance(vision_feature_layer, (list, tuple)) else [vision_feature_layer]
        self.hidden_size = hidden_size # LLM的特征维度
        self._attn_implementation_autoset = True # 防止transformers在创建模型时检查attn_implementation而本模型只支持原始attn而报错



class OmniPTQwen2VLConfig(Qwen2VLConfig):
    model_type = "omnipt_qwen2_vl"
    sub_configs = {"vision_config": Qwen2VLVisionConfig, "second_vision_config": OmniPTQwen2VLSecondVisionConfig}

    def __init__(
        self,
        second_vision_config=None,
        **kwargs,
    ):
        if isinstance(second_vision_config, dict):
            self.second_vision_config = self.sub_configs["second_vision_config"](**second_vision_config)
        elif second_vision_config is None:
            self.second_vision_config = self.sub_configs["second_vision_config"]()
        super().__init__(**kwargs)


OmniPTQwen2VLConfig.register_for_auto_class()
CONFIG_MAPPING.register(OmniPTQwen2VLConfig.model_type, OmniPTQwen2VLConfig)
__all__ = ["OmniPTQwen2VLSecondVisionConfig", "OmniPTQwen2VLConfig"]
