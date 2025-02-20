"""PyTorch OmniPath-LLaVA model."""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Dict, Any

import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.models.auto import AutoModel, AutoModelForCausalLM
from .configuration_omnipath import OmniPathConfig
from .module_mask import ResNetEncoder, MaskDecoderAllInOne, dice_loss


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "OmniPathConfig"


@dataclass
class OmniPathCausalLMOutputWithPast(ModelOutput):
    """
    Base class for OmniPath causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        masks (`tuple(torch.FloatTensor)`):
            Tuple of `torch.FloatTensor`, the tuplt has len=batch_size and the shape of each `torch.FloatTensor`
            is `(num_masks, 1, height, width)`.
            Results of the mask decoder.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)
            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`torch.FloatTensor`, *optional*):
            A `torch.FloatTensor` of size (num_images, image_length, embed_dim)`.
            image_hidden_states of the model produced by the vision encoder and after projecting.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    masks: Optional[Tuple[torch.FloatTensor]] = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[torch.FloatTensor] = None


class OmniPathMultiModalProjector(nn.Module):
    def __init__(self, config: OmniPathConfig):
        super().__init__()

        self.linear_1 = nn.Linear(config.vision_config.hidden_size, config.text_config.hidden_size, bias=True)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=True)

    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


OMNIPATH_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`OmniPathConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare OmniPath Model outputting raw hidden-states without any specific head on top.",
    OMNIPATH_START_DOCSTRING,
)
class OmniPathPreTrainedModel(PreTrainedModel):
    config_class = OmniPathConfig
    base_model_prefix = "language_model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlavaVisionAttention"]
    _skip_keys_device_placement = "past_key_values"
    _supports_cache_class = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def _init_weights(self, module):
        # important: this ported version of LLaVA isn't meant for training from scratch - only
        # inference and fine-tuning - so the proper init weights code has been removed - the original codebase
        # https://github.com/haotian-liu/LLaVA/tree/main/llava should serve for that purpose
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


OMNIPATH_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        pixel_values (List of `torch.FloatTensor` of shape `(num_images, num_channels, image_size, image_size)):
            The list has len `batch_size` and each element is a `torch.FloatTensor` of shape `(num_images, num_channels, image_size, image_size)`.
            The tensors corresponding to the input images. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`CLIPImageProcessor.__call__`] for details ([`LlavaProcessor`] uses
            [`CLIPImageProcessor`] for processing images).
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        vision_feature_layer (`int`, *optional*, defaults to -2):
            The index of the layer to select the vision feature.
        vision_feature_select_strategy (`str`, *optional*, defaults to `"default"`):
            The feature selection strategy used to select the vision feature from the vision backbone.
            Can be one of `"default"` or `"full"`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


@add_start_docstrings(
    """The OmniPath model which consists of a language model, a vision backbone, a mask encoder and decoder.""",
    OMNIPATH_START_DOCSTRING,
)
class OmniPathForConditionalGeneration(OmniPathPreTrainedModel, GenerationMixin):
    def __init__(self, config: OmniPathConfig):
        super().__init__(config)
        self.language_model = AutoModelForCausalLM.from_config(config.text_config)

        self.vision_tower = AutoModel.from_config(config.vision_config)
        self.multi_modal_projector = OmniPathMultiModalProjector(config)

        self.mask_encoder = ResNetEncoder(embed_dim=config.text_config.hidden_size)
        self.mask_decoder = MaskDecoderAllInOne(config)

        self.vocab_size = config.text_config.vocab_size
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.image_id = config.image_token_index
        self.mask_id = config.mask_token_index
        self.patch_id = config.patch_token_index

        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def tie_weights(self):
        return self.language_model.tie_weights()

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    def get_image_features(
        self, pixel_values: torch.FloatTensor, vision_feature_layer: int, vision_feature_select_strategy: str
    ):
        """
        Obtains image last hidden states from the vision tower and apply multimodal projection.

        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`)
               The tensors corresponding to the input images.
            vision_feature_layer (`int`):
                The index of the layer to select the vision feature.
            vision_feature_select_strategy (`str`):
                The feature selection strategy used to select the vision feature from the vision backbone.
                Can be one of `"default"` or `"full"`
        Returns:
            image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
        """
        image_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
        # this is not memory efficient at all (output_hidden_states=True) will save all the hidden stated.
        selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
        if vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        elif vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature
        else:
            raise ValueError(f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}")
        image_features = self.multi_modal_projector(selected_image_feature)
        return image_features
    
    def fill_vision_features(self, input_ids, inputs_embeds, vision_name, image_features):
        vision_id = getattr(self, f"{vision_name}_id")

        n_image_tokens = (input_ids == vision_id).sum().item()
        n_image_features = image_features.shape[0] * image_features.shape[1]

        if n_image_tokens != n_image_features:
            raise ValueError(
                f"{vision_name} features and {vision_name} tokens do not match: "
                f"tokens: {n_image_tokens}, features {n_image_features}"
            )
        special_image_mask = (
            (input_ids == vision_id)
            .unsqueeze(-1)
            .expand_as(inputs_embeds)
            .to(inputs_embeds.device)
        )
        image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
        return inputs_embeds.masked_scatter(special_image_mask, image_features)

    @torch.no_grad()
    def calculate_image_indices(self, input_ids: torch.Tensor):
        """
        Calculate sequential indices for image tokens across the entire batch of input sequences.
        
        Args:
            input_ids (torch.Tensor): Input tensor of shape (batch_size, seq_len) containing token IDs.
            
        Returns:
            torch.Tensor: Tensor of same shape as input_ids, where image tokens are 
                        assigned sequential indices starting from 1, continuing across
                        all sequences in the batch. Non-image tokens are set to 0.
        
        Note:
            - Image indices start from 1, with 0 reserved for non-image tokens
            - Indexing is continuous across the batch, e.g., if sequence 1 has images
            indexed 1-3, sequence 2's images will start from 4
        """
        image_mask = input_ids == self.image_id
        is_new_group = image_mask & ~torch.cat([
            torch.tensor([False]).expand(image_mask.shape[0], 1).to(image_mask.device), 
            image_mask[:, :-1]], dim=1)
        image_indices = torch.cumsum(is_new_group, dim=1)
        cumsum_indices = image_indices.max(dim=1, keepdim=True).values.cumsum(dim=0)[:-1]
        cumsum_indices = torch.cat([torch.tensor([[0]]).to(cumsum_indices.device), cumsum_indices], dim=0)
        image_indices = (image_indices + cumsum_indices) * image_mask
        return image_indices # Caution: Index starts at 1
    
    @torch.no_grad()
    def get_mask_info(self, id_mask, label_mask, input_ids):
        """
        Extract information about mask tokens and their corresponding image indices for generation.

        Args:
            id_mask (torch.Tensor): Binary <mask> tensor from input_ids indicating all mask for both encode and decode,
                                    shape: [batch_size, sequence_length]
            label_mask (torch.Tensor): Binary <mask> tensor from labels indicating mask for decode only,
                                        shape: [batch_size, sequence_length]  
            input_ids (torch.Tensor): Input token IDs, shape: [batch_size, sequence_length]

        Returns:
            batch_indices (torch.Tensor): Batch indices for each mask position
            mask_indices (torch.Tensor): Sequential indices (0-based) for masks within each sequence
            image_for_mask (torch.Tensor): Indices (0-based) of the most recent image token
                                            before each mask position

        Note:
            All returned indices are 0-based. For image_for_mask, it references the global 
            image numbering across the entire batch from calculate_image_indices().
        """
        mask_indices = id_mask.cumsum(dim=1) * label_mask # [batch_size, sequence_length]
        batch_indices, mask_positions = torch.where(mask_indices) # [number_of_mask_for_generation]
        mask_indices = mask_indices[batch_indices, mask_positions] - 1 # [number_of_mask_for_generation], start at 0
        image_indices = self.calculate_image_indices(input_ids)
        image_for_mask = torch.tensor([image_indices[x, :y].max() for x, y in zip(batch_indices, mask_positions)])
        image_for_mask = image_for_mask.to(label_mask.device) - 1 # [number_of_mask_for_generation], start at 0
        return batch_indices, mask_indices, image_for_mask

    @add_start_docstrings_to_model_forward(OMNIPATH_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=OmniPathCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: List[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[int] = None,
        vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        patch_pixel_values: Optional[List[torch.FloatTensor]] = None,
        masks: Optional[List[torch.FloatTensor]] = None,
        previous_last_hidden_states: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, OmniPathCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

            patch_pixel_values (List of `torch.FloatTensor`, *optional*):
                The list has len `batch_size` and each element is a `torch.FloatTensor` of shape `(num_images, num_channels, image_size, image_size)`.
                The tensors corresponding to the patches provided by the tool (function calling).

            masks (`List[torch.FloatTensor]`, *optional*):
                List of Tensors with shape [num_mask, 1, image_size, image_size], each data have different num_mask.
                The len of list is batch_size.
                The ground truth mask image for each <mask> token.

            previous_last_hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                The last hidden states from the previous decoding step. This is used for replace the input embeddings of the generated <mask> tokens. This is only used during generation.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, OmniPathForConditionalGeneration

        >>> model = OmniPathForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
        >>> processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

        >>> prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, text=prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_new_tokens=15)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "USER:  \nWhat's the content of the image? ASSISTANT: The image features a busy city street with a stop sign prominently displayed"
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # Encode images and masks
        if input_ids.shape[-1] == 1: # Generation
            # During generation, we use the generated embeddings of the previous step to replace the <mask> token's embedding.
            replace_mask = (input_ids == self.mask_id).flatten().to(device=input_ids.device)
            if replace_mask.sum() > 0:
                selected_pixel_values = [pixel_values[i][-1] for i in torch.where(replace_mask)[0]]
                selected_image_features = self.get_image_features(
                    pixel_values=torch.stack(selected_pixel_values, dim=0),
                    vision_feature_layer=vision_feature_layer,
                    vision_feature_select_strategy=vision_feature_select_strategy,
                )
                inputs_embeds[replace_mask, -1] = self.mask_encoder(self.mask_decoder(previous_last_hidden_states[replace_mask, -1:], selected_image_features))
        else: # Training
            # First encode the image features
            image_features = None
            if pixel_values is not None:
                image_features = self.get_image_features(
                    pixel_values=torch.cat(pixel_values, dim=0),
                    vision_feature_layer=vision_feature_layer,
                    vision_feature_select_strategy=vision_feature_select_strategy,
                )

            if image_features is not None:
                inputs_embeds = self.fill_vision_features(input_ids, inputs_embeds, 'image', image_features)

            # Then encode the patch features
            patch_features = None
            if patch_pixel_values is not None:
                patch_features = self.get_image_features(
                    pixel_values=torch.cat(patch_pixel_values, dim=0),
                    vision_feature_layer=vision_feature_layer,
                    vision_feature_select_strategy=vision_feature_select_strategy,
                )

            if patch_features is not None:
                inputs_embeds = self.fill_vision_features(input_ids, inputs_embeds, 'patch', patch_features)

            # During training, we encode the actual mask image to replace the <mask> token's embedding.
            mask_token_mask = (input_ids == self.mask_id).to(inputs_embeds.device)
            mask_valid_flag = masks is not None and any(len(m)>0 for m in masks)
            if mask_valid_flag or mask_token_mask.sum() > 0:
                assert all([len(m) == mask_token_mask[i].sum() for i, m in enumerate(masks)]), \
                    f'mask num not equal to <mask> token num: {[(len(m), mask_token_mask[i].sum()) for i, m in enumerate(masks)]}'
                mask_embeds = self.mask_encoder(torch.cat(masks, dim=0))
                inputs_embeds[mask_token_mask] = mask_embeds

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
        )

        logits = outputs.logits
        mask_pred = None

        loss = None
        if labels is not None:
            # Calculate the token loss.
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_last_hidden_states = outputs.hidden_states[-1][..., :-1, :].contiguous()
            # Flatten the tokens and calculate the loss
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

            # Calculate the mask prediction loss
            if mask_valid_flag:
                mask_token_mask = mask_token_mask.to(logits.device)
                mask_label_mask = (labels == self.mask_id).to(logits.device)
                mask_hs = shift_last_hidden_states[mask_label_mask[..., 1:]].unsqueeze(1) # [number of <mask>, 1, D]
                batch_indices, mask_indices, image_for_mask = self.get_mask_info(mask_token_mask, mask_label_mask, input_ids)
                mask_features = torch.stack([image_features[i] for i in image_for_mask], dim=0).to(mask_hs.device)
                # [num_mask, 1, D] [num_mask, HW, D] => [num_mask, 1, H, W]
                mask_pred = self.mask_decoder(mask_hs, mask_features)
                mask_gt = torch.stack([masks[i][j] for i, j in zip(batch_indices, mask_indices)], dim=0).to(mask_pred.device)
                loss += self.mask_loss(mask_pred, mask_gt)
                batch_mask_num = [0] * labels.shape[0]
                for b in batch_indices:
                    batch_mask_num[b] += 1
                mask_pred = torch.split(mask_pred, batch_mask_num, dim=0)
        elif input_ids.shape[-1] != 1:
            # When no labels are provided, we generate the masks according to the logits.
            predictions = logits.argmax(dim=-1)
            last_hidden_states = outputs.hidden_states[-1]
            mask_token_mask = (predictions == self.mask_id).to(logits.device)
            batch_mask_num = mask_token_mask.sum(dim=1).tolist()
            mask_hs = last_hidden_states[mask_token_mask].unsqueeze(1) # [number of <mask>, 1, D]
            batch_indices, mask_indices, image_for_mask = self.get_mask_info(mask_token_mask, mask_token_mask, input_ids)
            mask_features = torch.stack([image_features[i] for i in image_for_mask], dim=0).to(mask_hs.device)
            mask_pred = self.mask_decoder(mask_hs, mask_features)
            mask_pred = torch.split(mask_pred, batch_mask_num, dim=0)

        output.hidden_states = output.hidden_states if output_hidden_states else None

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        return OmniPathCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            masks=mask_pred,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )
    
    def mask_loss(self, m_pred, m_true):
        loss_dice = dice_loss(F.sigmoid(m_pred.squeeze(1)), m_true.squeeze(1).float(), multiclass=False)
        m_pred = m_pred.view(-1)
        m_true = m_true.view(-1)
        loss_bce = F.binary_cross_entropy_with_logits(m_pred, m_true)
        return loss_dice + loss_bce

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        attention_mask=None,
        cache_position=None,
        num_logits_to_keep=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        model_inputs = self.language_model.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
            **kwargs,
        )
        model_inputs["pixel_values"] = pixel_values

        return model_inputs
    
    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        """Function that is used after forward in generate to process model inputs for next step forward."""

        #This function updates the model's keyword arguments for generation tasks by handling the caching of past key-values, extending attention masks, and managing token type IDs. For each new token to be generated, it ensures all necessary tensors are properly extended to accommodate the new position, supporting both encoder-decoder and decoder-only architectures.
        model_kwargs = self.language_model._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=is_encoder_decoder,
            standardize_cache_format=standardize_cache_format,
            num_new_tokens=num_new_tokens,
        )
        model_kwargs['previous_last_hidden_states'] = outputs.hidden_states[-1].detach().clone()
        
        return model_kwargs