import timm
import torch
from functools import partial
from typing import Optional, List, Tuple, Union
from timm.models.vision_transformer import VisionTransformer, Block
from transformers.models.qwen2_vl.modeling_qwen2_vl import PatchMerger, Qwen2VLPreTrainedModel, Qwen2VLForConditionalGeneration, Qwen2VLCausalLMOutputWithPast
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLVisionConfig
from transformers import Qwen2VLImageProcessor
from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING

from .configuration_omnipt_qwen2_vl import OmniPTQwen2VLSecondVisionConfig, OmniPTQwen2VLConfig


def create_timm_vit_from_config(config: OmniPTQwen2VLSecondVisionConfig) -> VisionTransformer:
    timm_kwargs = {
        'model_name': config.model_name,
        'pretrained': False,
        'num_classes': 0,
        'dynamic_img_size': True,
        'dynamic_img_pad': True,
    }
    if 'paige-ai/Virchow' in config.model_name: # "hf-hub:paige-ai/Virchow" and "hf-hub:paige-ai/Virchow2"
        return timm.create_model(mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU, **timm_kwargs)
    elif config.model_name == "hf-hub:MahmoodLab/uni":
        return timm.create_model(init_values=1e-5, **timm_kwargs)
    elif config.model_name == "hf-hub:MahmoodLab/UNI2-h":
        timm_kwargs.update({
            'img_size': 224,
            'patch_size': 14,
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5,
            'embed_dim': 1536,
            'mlp_ratio': 2.66667*2,
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked,
            'act_layer': torch.nn.SiLU,
            'reg_tokens': 8,
        })
        return timm.create_model(**timm_kwargs)
    else:
        return timm.create_model(**timm_kwargs)


class OmniPTQwen2TimmViTPretrainedModel(Qwen2VLPreTrainedModel):
    config_class = OmniPTQwen2VLSecondVisionConfig
    supports_gradient_checkpointing = False # VisionTransformer类内部支持gradient_checkpointing，但是forward_intermediates不支持
    _no_split_modules = [Block]
    _supports_flash_attn_2 = False
    _supports_sdpa = False

    def __init__(self,
                 config: OmniPTQwen2VLSecondVisionConfig,
                 first_vision_config: Qwen2VLVisionConfig,
                 image_processor: Qwen2VLImageProcessor
    ) -> None:
        super().__init__(config)
        self.fvc = first_vision_config
        self.img_proc = image_processor

        self.vit = create_timm_vit_from_config(config)
        self.embed_dim = self.vit.embed_dim * len(config.vision_feature_layer)
        vision_feature_layer = [len(self.vit.blocks)+l if l < 0 else l for l in config.vision_feature_layer]
        self.vit.forward = partial(
            self.vit.forward_intermediates,
            indices=vision_feature_layer,
            return_prefix_tokens=False,
            norm=False,
            stop_early=True,
            output_fmt='NLC',
            intermediates_only=True,
        )
        assert self.fvc.patch_size == self.vit.patch_embed.patch_size[0], \
            f"First vision encoder patch size {self.fvc.patch_size} does not match " \
            f"the second vision encoder patch size {self.vit.patch_embed.patch_size[0]}."

        self.merger = PatchMerger(
            dim=config.hidden_size,
            context_dim=self.embed_dim,
            spatial_merge_size=self.fvc.spatial_merge_size
        )

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)

        old_mean = torch.tensor(self.img_proc.image_mean, dtype=pixel_values.dtype, device=pixel_values.device).view(1, -1, 1, 1)
        old_std = torch.tensor(self.img_proc.image_std, dtype=pixel_values.dtype, device=pixel_values.device).view(1, -1, 1, 1)
        new_mean = torch.tensor(self.vit.pretrained_cfg['mean'], dtype=pixel_values.dtype, device=pixel_values.device).view(1, -1, 1, 1)
        new_std = torch.tensor(self.vit.pretrained_cfg['std'], dtype=pixel_values.dtype, device=pixel_values.device).view(1, -1, 1, 1)

        image_features = []
        for i, (grid_t, grid_h, grid_w) in enumerate(grid_thw):
            assert grid_t == 1, f'grid_t == 1 is required for OmniPTQwen2TimmViTPretrainedModel, but got {grid_t = }'

            # 1. 从Qwen的图像格式中恢复符合timm的图像格式[B,C,H,W]
            pixel_value = pixel_values[cu_seqlens[i]: cu_seqlens[i+1]]
            pixel_value = pixel_value.view(
                grid_t, grid_h // self.fvc.spatial_merge_size, grid_w // self.fvc.spatial_merge_size,
                self.fvc.spatial_merge_size, self.fvc.spatial_merge_size,
                self.fvc.in_channels, self.fvc.temporal_patch_size, self.fvc.patch_size, self.fvc.patch_size
            ).permute(0, 6, 5, 1, 3, 7, 2, 4, 8)[:, 0] # temporal_patch_size=2为了适配Conv3d会复制一份图像，因此取前面的就好
            pixel_value = pixel_value.reshape(grid_t, self.fvc.in_channels, grid_h*self.fvc.patch_size, grid_w*self.fvc.patch_size)
            
            # 2. 对像素值范围进行变换
            pixel_value = pixel_value * old_std + old_mean
            pixel_value = (pixel_value - new_mean) / new_std

            # 3. 获取图像特征
            image_feature = torch.cat(self.vit(pixel_value), dim=-1) # [B, L, C]
            image_feature = image_feature.view(
                grid_t, grid_h // self.fvc.spatial_merge_size, self.fvc.spatial_merge_size,
                grid_w // self.fvc.spatial_merge_size, self.fvc.spatial_merge_size, self.embed_dim
            ).permute(0, 1, 3, 2, 4, 5).reshape(-1, self.embed_dim)
            image_features.append(image_feature)

        image_features = torch.cat(image_features, dim=0)
        return self.merger(image_features)
    
    def get_dtype(self) -> torch.dtype:
        return self.vit.blocks[0].mlp.fc2.weight.dtype

    def get_device(self) -> torch.device:
        return self.vit.blocks[0].mlp.fc2.weight.device


class OmniPTQwen2VLForConditionalGeneration(Qwen2VLForConditionalGeneration):
    config_class = OmniPTQwen2VLConfig

    def __init__(self, config: OmniPTQwen2VLConfig):
        super().__init__(config)

        self.second_visual = OmniPTQwen2TimmViTPretrainedModel._from_config(
            config.second_vision_config,
            first_vision_config = config.vision_config,
            image_processor = Qwen2VLImageProcessor.from_pretrained(self.name_or_path)
        )

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, Qwen2VLCausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                image_embeds = self.visual(pixel_values.type(self.visual.get_dtype()), grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )
                image_embeds2 = self.second_visual(pixel_values.type(self.second_visual.get_dtype()), grid_thw=image_grid_thw)
                if image_embeds.shape != image_embeds2.shape:
                    raise ValueError(
                        f"The origin and second image features do not match: {image_embeds.shape=}, {image_embeds2.shape=}"
                    )
                image_embeds = image_embeds + image_embeds2
                image_mask = (
                    (input_ids == self.config.image_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )
                video_mask = (
                    (input_ids == self.config.video_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids, image_grid_thw, video_grid_thw, attention_mask
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                    delta = delta.to(position_ids.device)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )
    

OmniPTQwen2VLForConditionalGeneration.register_for_auto_class('AutoModelForImageTextToText')
MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING.register(OmniPTQwen2VLConfig, OmniPTQwen2VLForConditionalGeneration)
__all__ = ["OmniPTQwen2TimmViTPretrainedModel", "OmniPTQwen2VLForConditionalGeneration"]
