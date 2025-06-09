import re
import math
from typing import Tuple, Type

import cv2
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F


DEFAULT_MASK_TOKEN = "<mask>"


# -------------------------- resnet mask encoder --------------------------

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


norm_map = {
    "ln": LayerNorm2d,
    "bn": nn.BatchNorm2d
}


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, norm="ln"):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm1 = norm_map[norm](out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, bias=False, padding=1)
        self.norm2 = norm_map[norm](out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.norm3 = norm_map[norm](out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)
        out = self.relu(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)

        out = shortcut + out
        out = self.relu(out)

        return out


class ResNetEncoder(nn.Module):
    def __init__(self, embed_dim, downblock=Bottleneck, num_layers=[2,2,2,2], norm_enc="ln"):
        super(ResNetEncoder, self).__init__()

        self.in_channels = 32

        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.norm1 = norm_map[norm_enc](32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.dlayer1 = self._make_downlayer(downblock, 32, num_layers[0], norm=norm_enc)
        self.dlayer2 = self._make_downlayer(downblock, 32, num_layers[1],
                                            stride=2, norm=norm_enc)
        self.dlayer3 = self._make_downlayer(downblock, 32, num_layers[2],
                                            stride=2, norm=norm_enc)
        self.dlayer4 = self._make_downlayer(downblock, 32, num_layers[3],
                                            stride=2, norm=norm_enc)
        self.avg_pool = nn.AdaptiveAvgPool2d(11)
        self.linear_enc = nn.Linear(64*11*11, 512)
        self.linear_dec = nn.Linear(512, embed_dim)


    def _make_downlayer(self, block, init_channels, num_layer, stride=1, norm="ln"):
        downsample = None
        norm_ = norm_map[norm]
        if stride != 1 or self.in_channels != init_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, init_channels*block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_(init_channels*block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, init_channels, stride, downsample, norm))
        self.in_channels = init_channels * block.expansion
        for i in range(1, num_layer):
            layers.append(block(self.in_channels, init_channels))

        return nn.Sequential(*layers)

    def encode(self, x):
        x_size = x.size()
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.dlayer1(x)
        x = self.dlayer2(x)
        x = self.dlayer3(x)
        x = self.dlayer4(x)
        x = self.avg_pool(x)
        x = self.linear_enc(x.reshape(x_size[0], -1))
        return x

    def forward(self, x):
        repr = self.encode(x)
        repr = self.linear_dec(repr)
        return repr


# -------------------------- sam mask decoder --------------------------

def conv_layer(in_dim, out_dim, kernel_size=1, padding=0, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        LayerNorm2d(out_dim), nn.ReLU(True))


class Projector(nn.Module):
    def __init__(self, in_dim=256):
        super().__init__()
        self.in_dim = in_dim
        # visual projector
        self.layer1 = conv_layer(in_dim, in_dim//4, 3, padding=1)
        self.layer2 = conv_layer(in_dim//4, in_dim//8, 3, padding=1)
        self.layer3 = conv_layer(in_dim//8, in_dim//8, 3, padding=1)
        self.conv = nn.Conv2d(in_dim//8, in_dim//8, 1)

    def forward(self, x):
        x = self.layer1(x)
        with torch.cuda.amp.autocast(enabled=False):
            x = torch.nn.functional.interpolate(x.to(torch.float32), scale_factor=2, mode="bilinear")
        x = self.layer2(x)
        with torch.cuda.amp.autocast(enabled=False):
            x = torch.nn.functional.interpolate(x.to(torch.float32), scale_factor=2, mode="bilinear")
        x = self.layer3(x)
        with torch.cuda.amp.autocast(enabled=False):
            x = torch.nn.functional.interpolate(x.to(torch.float32), scale_factor=2, mode="bilinear")
        x = self.conv(x)
        return x
    

class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int,
            sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


def pos2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(
        torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(
        0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(
        0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(
        0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(
        0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe.reshape(-1, height , width).cuda()# c, h, w


class MaskDecoder(nn.Module):
    def __init__(
            self,
            *,
            transformer_dim: int,
            transformer: nn.Module,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.output_upscaling = Projector(transformer_dim)
        self.output_hypernetworks_mlp = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)

    def forward(
            self,
            image_embeddings: torch.Tensor,
            prompt_embeddings: torch.Tensor,
    ):
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        b, h, w, c = image_embeddings.shape
        image_pe = pos2d(c, h, w)
        masks = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            prompt_embeddings=prompt_embeddings,
        )

        return masks

    def predict_masks(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        # Expand per-image data in batch direction to be per-mask
        src = image_embeddings.permute(0,3,1,2)
        pos_src = torch.repeat_interleave(image_pe.unsqueeze(0), src.shape[0], dim=0) #.permute(0,3,2,1) comment by ZSXM
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, prompt_embeddings)

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in = self.output_hypernetworks_mlp(hs)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        return masks


class VisualEmbedder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.shortcut = nn.Linear(input_dim, output_dim)
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.linear2 = nn.Linear(output_dim, output_dim)

    def forward(self, feat):
        out = self.linear2(F.relu(self.linear1(feat)))
        out = self.shortcut(feat) + out
        return out


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
            self,
            embedding_dim: int,
            num_heads: int,
            downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        # debug
        inf_value = torch.finfo(attn.dtype).max
        neg_inf_value = -inf_value
        attn[attn == float('inf')] = inf_value
        attn[attn == float('-inf')] = neg_inf_value
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out
    

class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class TwoWayAttentionBlock(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            num_heads: int,
            mlp_dim: int = 2048,
            activation: Type[nn.Module] = nn.ReLU,
            attention_downsample_rate: int = 2,
            skip_first_layer_pe: bool = False,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
            self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class TwoWayTransformer(nn.Module):
    def __init__(
            self,
            depth: int,
            embedding_dim: int,
            num_heads: int,
            mlp_dim: int,
            activation: Type[nn.Module] = nn.ReLU,
            attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
            self,
            image_embedding: Tensor,
            image_pe: Tensor,
            point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


# This class is add for directly predicting masks using <mask> token embedding and image features
class MaskDecoderAllInOne(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.image_size = config.vision_config.image_size
        self.decoder_dim = 1024
        self.patch_edge = self.image_size // config.vision_config.patch_size
        self.decoder_mapping = nn.Linear(config.text_config.hidden_size, self.decoder_dim)
        self.visual_adaptor = VisualEmbedder(config.vision_config.hidden_size, self.decoder_dim)
        self.mask_decoder = MaskDecoder(
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.decoder_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.decoder_dim,
        )

    def forward(self, mask_embedding, image_feature):
        """
        Args:
          mask_embedding (torch.Tensor): the token embedding of <mask> generated by LLM with shape B x N_points x embedding_dim for any N_points.
          image_feature (torch.Tensor): image features to attend to. Should be shape
            B x HW x embedding_dim for any H and W.

        Returns:
          mask with shape B x N_points x H x W
        """
        mask_embeddings_mapped = self.decoder_mapping(mask_embedding)
        image_feature = self.visual_adaptor(image_feature).reshape(image_feature.size(0), self.patch_edge, self.patch_edge, self.decoder_dim)
        masks = self.mask_decoder(image_feature, mask_embeddings_mapped)
        masks = F.interpolate(masks, self.image_size, mode="bilinear")
        return masks


# -------------------------- mask ops --------------------------
def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


def find_white_regions(image, max_vertices=50):
    """
    Find white regions in a binary image and return their bounding boxes and boundary polygons.
    This version adjusts epsilon dynamically to achieve the desired number of vertices for polygons.

    :param image: A single-channel binary image.
    :param max_vertices: Maximum number of vertices for the boundary polygon.
    :return: Two lists - one for bounding boxes and one for boundary polygons.
    """
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = []
    boundary_polygons = []

    # Check if there are no contours found
    if not contours:
        return bounding_boxes, boundary_polygons  # Return empty lists if no contours found

    img_height, img_width = image.shape

    for contour in contours:
        # Compute the bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append([round(x / img_width, 3), round(y / img_height, 3), 
                               round((x + w) / img_width, 3), round((y + h) / img_height, 3)])

        # Dynamically adjust epsilon to reduce the number of vertices
        epsilon = 0.001 * cv2.arcLength(contour, True)  # initial epsilon
        while True:
            approx_polygon = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx_polygon) <= max_vertices:
                break
            epsilon *= 1.1  # Increase epsilon

        # Normalize the coordinates of the polygon points and add to the list
        polygon = [[round(point[0][0] / img_width, 3), round(point[0][1] / img_height, 3)] for point in approx_polygon]
        boundary_polygons.append(polygon)

    # Sorting the regions based on the top-left corner of bounding boxes
    sorted_combined = sorted(zip(bounding_boxes, boundary_polygons), key=lambda x: (x[0][0], x[0][1]))

    # Unzipping the sorted pairs
    bounding_boxes_sorted, boundary_polygons_sorted = map(list, zip(*sorted_combined)) if sorted_combined else ([], [])

    return bounding_boxes_sorted, boundary_polygons_sorted


def extract_and_replace_masks(contour_str, image_size, contour_tag='contour_list', polygon_tag='polygon', replacement_tag=DEFAULT_MASK_TOKEN):
    """
    输入： 
    - contour_str: 包含<contour_tag>和<polygon_tag>的字符串
    - image_size: 图像尺寸（宽, 高）
    - contour_tag: 标签用于包裹多边形的外部标签名称
    - polygon_tag: 标签用于包裹单个多边形的标签名称
    - replacement_tag: 用于替换contour_tag的标签
    
    返回：
    - new_contour_str: 替换了contour_tag为replacement_tag的字符串
    - masks: 包含每个contour_tag对应的掩码图像列表
    """
    # 定义正则表达式，提取多个contour_tag块
    contour_pattern = f"<{contour_tag}>.*?</{contour_tag}>"
    
    # 提取所有contour_tag块
    contour_blocks = re.findall(contour_pattern, contour_str, re.DOTALL)
    
    # 定义正则表达式，提取多边形中的坐标，兼容圆括号和方括号
    polygon_pattern = f"<{polygon_tag}>[\\(\\[].*?[\\)\\]]</{polygon_tag}>"
    
    # 初始化空的掩码图像列表
    masks = []

    # 替换后的字符串初始化为原始的contour_str
    new_contour_str = contour_str

    # 对每个contour_tag块进行处理
    for block in contour_blocks:
        # 初始化一个纯黑图像
        mask = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
        
        # 提取该块中的所有多边形
        polygons = re.findall(polygon_pattern, block)
        
        # 解析并处理每个多边形
        for polygon in polygons:
            # 提取坐标对
            coords = re.findall(r'\(?(\d*\.\d+|\d+),\s*(\d*\.\d+|\d+)\)?', polygon)
            
            # 转换为实际坐标值
            points = [(int(float(x) * image_size[0]), int(float(y) * image_size[1])) for x, y in coords]
            
            # 转换为符合OpenCV格式的 numpy 数组
            points_array = np.array([points], dtype=np.int32)
            
            # 在mask图像上填充绘制多边形
            cv2.fillPoly(mask, points_array, 255)
        
        # 将该块的mask添加到结果列表中
        masks.append(mask)
        
        # 替换该块为新的replacement_tag
        new_contour_str = new_contour_str.replace(block, replacement_tag)
    
    return new_contour_str, masks


def restore_masks(new_contour_str, masks, replacement_tag=DEFAULT_MASK_TOKEN, contour_tag='contour_list', polygon_tag='polygon'):
    """
    输入：
    - new_contour_str: 包含replacement_tag标签的字符串
    - masks: 掩码图像列表
    - replacement_tag: 标签用于替换回原始的contour_tag
    - contour_tag: 用于还原的多边形外部标签名称
    - polygon_tag: 用于还原的多边形标签名称
    
    返回：
    - contour_str: 还原后的包含<contour_tag>和<polygon_tag>的字符串
    """
    # 初始化contour_str为新的输入字符串
    contour_str = new_contour_str
    
    # 使用cv2.findContours从mask中提取轮廓
    for mask in masks:
        # 提取轮廓
        _, contours = find_white_regions(mask)
        
        # 初始化一个contour_list块
        contour_block = f"<{contour_tag}>"
        
        # 处理每个轮廓
        for contour in contours:
            # 初始化一个polygon块
            polygon_block = f"<{polygon_tag}>"
            
            # 将轮廓坐标转换为相对坐标（比例形式）
            for x_rel, y_rel in contour:
                # 将坐标加入polygon块
                polygon_block += f"[{x_rel:.3f}, {y_rel:.3f}], "
            
            # 去掉最后一个逗号和空格
            polygon_block = polygon_block.rstrip(", ")
            polygon_block += f"</{polygon_tag}>"
            
            # 将polygon块加入contour_block
            contour_block += polygon_block
        
        # 完成contour_block
        contour_block += f"</{contour_tag}>"
        
        # 将 <replacement_tag> 替换为还原的 contour_block
        contour_str = contour_str.replace(replacement_tag, contour_block, 1)
    
    return contour_str
