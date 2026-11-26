from functools import partial
from pathlib import Path
from typing import Optional, Union

import torch
from torch.nn import functional as F

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from .sam import Sam
from .transformer import TwoWayTransformer

__copyright__ = "Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved."
__license__ = "Apache-2.0"
__notice__ = "Modified for WILD-SAM. See THIRD_PARTY_NOTICES.md."


def _load_backbone_weights(sam: Sam, checkpoint: Union[str, Path]) -> None:
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    expected = sam.state_dict()
    for name, tensor in list(weights.items()):
        if name not in expected or not isinstance(tensor, torch.Tensor):
            continue
        target = expected[name]
        if tensor.shape == target.shape:
            continue
        if name == "image_encoder.pos_embed" and tensor.ndim == 4:
            resized = F.interpolate(
                tensor.permute(0, 3, 1, 2).contiguous(),
                size=target.shape[1:3],
                mode="bicubic",
                align_corners=False,
            )
            weights[name] = resized.permute(0, 2, 3, 1).contiguous()
        elif name.endswith((".attn.rel_pos_h", ".attn.rel_pos_w")) and tensor.ndim == 2:
            resized = F.interpolate(
                tensor.T.unsqueeze(0).contiguous(),
                size=target.shape[0],
                mode="linear",
                align_corners=False,
            )
            weights[name] = resized.squeeze(0).T.contiguous()
    sam.load_state_dict(weights, strict=True)


def build_sam_vit_b(
    image_size: int = 256,
    checkpoint: Optional[Union[str, Path]] = None,
) -> Sam:
    if image_size < 16 or image_size % 16:
        raise ValueError("image_size must be a positive multiple of 16.")
    prompt_embed_dim = 256
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=12,
            embed_dim=768,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=12,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=(2, 5, 8, 11),
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    if checkpoint is not None:
        _load_backbone_weights(sam, checkpoint)
    return sam.eval()
