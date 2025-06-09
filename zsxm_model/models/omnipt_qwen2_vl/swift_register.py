from PIL import Image
from typing import Dict, Any
from swift.llm import register_model, Model, ModelGroup, ModelMeta, get_model_tokenizer_multimodal
from swift.llm import register_model_arch, MultiModelKeys, register_template, to_device
from swift.llm.model.patcher import patch_output_clone, patch_output_to_input_device
from swift.llm.model.model.qwen import patch_qwen_vl_utils
from swift.llm.template.template.qwen import QwenTemplateMeta, Qwen2VLTemplate
from swift.utils import is_deepspeed_enabled


# ==================== OmniPT-Qwen2-VL ====================
class OmniPTQwen2VLTemplate(Qwen2VLTemplate):
    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_training:
            return inputs
        input_ids = inputs['input_ids']
        _model = model.model
        if not hasattr(_model, 'embed_tokens'):
            _model = _model.model  # LoRA
        pixel_values = inputs.get('pixel_values')
        pixel_values_videos = inputs.get('pixel_values_videos')
        image_grid_thw = inputs.get('image_grid_thw')
        video_grid_thw = inputs.get('video_grid_thw')

        inputs_embeds = _model.embed_tokens(input_ids)

        dtype = model.visual.get_dtype() if self.version == 'v2' else model.visual.dtype
        if pixel_values is None and pixel_values_videos is None:  # plain-text
            if is_deepspeed_enabled():
                images = [Image.new('RGB', (32, 32), (0, 0, 0))]
                media_inputs = self.processor.image_processor(images=images, videos=None, return_tensors='pt')
                device = input_ids.device
                media_inputs = to_device(media_inputs, device)
                pixel_values = media_inputs['pixel_values'].type(dtype)
                image_embeds = model.visual(pixel_values, grid_thw=media_inputs['image_grid_thw'])
                image_embeds2 = model.second_visual(pixel_values, grid_thw=media_inputs['image_grid_thw'])
                image_embeds = image_embeds + image_embeds2
                inputs_embeds += image_embeds.mean() * 0.
        else:
            if pixel_values is not None:
                pixel_values = pixel_values.type(dtype)
                image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw)
                image_embeds2 = model.second_visual(pixel_values, grid_thw=image_grid_thw)
                image_embeds = image_embeds + image_embeds2
                image_mask = (input_ids == model.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(dtype)
                video_embeds = model.visual(pixel_values_videos, grid_thw=video_grid_thw)
                video_mask = (input_ids == model.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        return {'inputs_embeds': inputs_embeds}

register_template(QwenTemplateMeta('omnipt_qwen2_vl', template_cls=OmniPTQwen2VLTemplate))


register_model_arch(
    MultiModelKeys(
        'omnipt_qwen2_vl',
        language_model='model',
        aligner=['visual.merger', 'second_visual.merger'],
        vision_tower=['visual', 'second_visual'],
    ))


def get_model_tokenizer_omnipt_qwen2_vl(*args, **kwargs):
    from zsxm_model.models.omnipt_qwen2_vl import OmniPTQwen2VLForConditionalGeneration
    kwargs['automodel_class'] = kwargs['automodel_class'] or OmniPTQwen2VLForConditionalGeneration
    model, tokenizer = get_model_tokenizer_multimodal(*args, **kwargs)
    if model is not None and hasattr(model.model, 'embed_tokens'):
        patch_output_clone(model.model.embed_tokens)
        patch_output_to_input_device(model.model.embed_tokens)

    from qwen_vl_utils import vision_process
    patch_qwen_vl_utils(vision_process)
    return model, tokenizer

register_model(
    ModelMeta(
        'omnipt_qwen2_vl',
        [
            ModelGroup([
                Model(model_path='/c22073/LLM_weights/OmniPT-Qwen2-VL-7B'),
            ]),
        ],
        template='omnipt_qwen2_vl',
        get_function=get_model_tokenizer_omnipt_qwen2_vl,
        model_arch='omnipt_qwen2_vl',
        architectures=['OmniPTQwen2VLForConditionalGeneration'],
        requires=['transformers>=4.45', 'qwen_vl_utils>=0.0.6', 'decord'],
        tags=['vision', 'video'],
        is_multimodal=True,
    )
)