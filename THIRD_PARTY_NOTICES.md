# Third-party notices

The files in `wild_sam/modeling/sam/` derive from Meta's Segment Anything
implementation. Copyright (c) Meta Platforms, Inc. and affiliates. They are
distributed under the Apache License 2.0. The Python modules carry copyright and license metadata. Additional source attributions appear below.

WILD-SAM adapts the ViT-B builder to the interferogram input size, interpolates
absolute and relative position embeddings when loading a SAM checkpoint, and
extends the mask decoder to accept a batch of images with paired prompts.

- [Segment Anything source at the referenced revision](https://github.com/facebookresearch/segment-anything/tree/dca509fe793f601edb92606367a655c15ac00fdf)
- [Segment Anything license](licenses/segment-anything-LICENSE)

## Additional source attributions

- `common.py` uses LayerNorm2d adapted from [Detectron2](https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py) and [ConvNeXt](https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119).
- `image_encoder.py` adapts the [ViTDet backbone](https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py) and [MViT relative-position code](https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py).
- `mask_decoder.py` includes an MLP adapted from [MaskFormer](https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py).
