# WILD-SAM

PyTorch implementation of *WILD-SAM: Phase-Aware Expert Adaptation of SAM for Landslide Detection in Wrapped InSAR Interferograms*.

## Installation

Requires Python 3.9 or later.

```bash
python -m pip install .
```

## Usage

This example runs a forward pass with randomly initialized weights and a single foreground point in pixel coordinates.

```python
import math

import torch
from wild_sam import WILDSAM

model = WILDSAM().eval()
phase = torch.linspace(-math.pi, math.pi, 256).expand(256, 256)
channels = torch.stack((
    (phase + math.pi) / (2 * math.pi),
    (phase.sin() + 1) / 2,
    (phase.cos() + 1) / 2,
))
image = channels.unsqueeze(0) * 255
coords = torch.tensor([[[128., 96.]]])
labels = torch.tensor([[1]])

with torch.inference_mode():
    probability = model(image, coords, labels)["out"]
```

The output is a foreground probability map of shape `[1, 1, 256, 256]`. To initialize the SAM components from ViT-B weights, pass `sam_checkpoint="sam_vit_b_01ec64.pth"` to `WILDSAM`.

The SAM components are adapted from Meta's Segment Anything. [Third-party notices](THIRD_PARTY_NOTICES.md)
