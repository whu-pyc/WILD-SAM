from pathlib import Path
from typing import Optional, Union

import torch
from torch import Tensor, nn

from .modeling.pa_moe import PAMoEAdapter
from .modeling.sam import build_sam_vit_b
from .modeling.wgse import WGSE


ADAPTED_BLOCKS = {
    "S": (1, 6, 12),
    "B": (2, 4, 6, 8, 10, 12),
    "L": tuple(range(1, 13)),
}


class WILDSAM(nn.Module):

    def __init__(self, variant: str = "L", sam_checkpoint: Optional[Union[str, Path]] = None):
        super().__init__()
        if variant not in ADAPTED_BLOCKS:
            raise ValueError("variant must be 'S', 'B', or 'L'")
        self.variant = variant
        self.sam = build_sam_vit_b(image_size=256, checkpoint=sam_checkpoint)
        self.sam.image_encoder.requires_grad_(False)
        for block_id in ADAPTED_BLOCKS[variant]:
            attention = self.sam.image_encoder.blocks[block_id - 1].attn
            attention.qkv = PAMoEAdapter(attention.qkv)
        self.wgse = WGSE(channels=self.sam.prompt_encoder.embed_dim)

    def forward(
        self,
        image: Tensor,
        point_coords: Optional[Tensor] = None,
        point_labels: Optional[Tensor] = None,
        boxes: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        if image.ndim != 4 or image.size(1) != 3:
            raise ValueError("image must have shape [B, 3, H, W]")
        batch, _, height, width = image.shape
        if not (0 < height <= 256 and 0 < width <= 256):
            raise ValueError("resize the input and its prompt coordinates to fit 256 × 256")
        if (point_coords is None) != (point_labels is None):
            raise ValueError("provide point_coords and point_labels together")
        points = None
        if point_coords is not None:
            if (point_coords.ndim != 3 or point_coords.shape[0] != batch
                    or point_coords.shape[-1] != 2 or point_coords.shape[:2] != point_labels.shape):
                raise ValueError("expected point_coords [B, N, 2] and point_labels [B, N]")
            points = (point_coords.to(image.device, torch.float32), point_labels.to(image.device, torch.long))
        if boxes is None and points is None:
            boxes = image.new_tensor([0, 0, width - 1, height - 1], dtype=torch.float32).expand(batch, 4)
        if boxes is not None:
            if boxes.shape not in ((batch, 4), (batch, 1, 4)):
                raise ValueError("expected one box per image, of shape [B, 4] or [B, 1, 4]")
            boxes = boxes.to(image.device, torch.float32).reshape(batch, 1, 4)

        embedding = self.sam.image_encoder(self.sam.preprocess(image))
        embedding = self.wgse(embedding)
        sparse, dense = self.sam.prompt_encoder(points=points, boxes=boxes, masks=None)
        logits, _ = self.sam.mask_decoder(
            image_embeddings=embedding,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        logits = self.sam.postprocess_masks(logits, (height, width), (height, width))
        return {"out": logits.sigmoid()}
