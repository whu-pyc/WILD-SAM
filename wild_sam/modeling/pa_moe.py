import torch
from torch import Tensor, nn
from torch.nn import functional as F


class DepthwiseSeparableExpert(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.pointwise = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, features: Tensor) -> Tensor:
        return F.gelu(self.pointwise(self.depthwise(features)))


class DilatedExpert(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, 3, dilation=2, padding=2, groups=channels, bias=False
        )
        self.pointwise = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, features: Tensor) -> Tensor:
        return F.gelu(self.pointwise(self.depthwise(features)))


class AsymmetricExpert(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.horizontal = nn.Conv2d(
            channels, channels, (1, 7), padding=(0, 3), groups=channels, bias=False
        )
        self.vertical = nn.Conv2d(
            channels, channels, (7, 1), padding=(3, 0), groups=channels, bias=False
        )
        self.pointwise = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, features: Tensor) -> Tensor:
        directional = self.vertical(self.horizontal(features))
        return F.gelu(self.pointwise(directional))


class HighFrequencyExpert(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        kernel = torch.tensor([[0., -1., 0.], [-1., 4., -1.], [0., -1., 0.]])
        self.register_buffer(
            "laplacian", kernel.view(1, 1, 3, 3).repeat(channels, 1, 1, 1), persistent=False
        )
        self.pointwise = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, features: Tensor) -> Tensor:
        edges = F.conv2d(features, self.laplacian.to(features.dtype), padding=1, groups=features.size(1))
        return F.gelu(self.pointwise(edges))


class DynamicRouter(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        hidden = max(16, channels // 4)
        self.mlp = nn.Sequential(nn.Linear(channels, hidden), nn.GELU(), nn.Linear(hidden, 4))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, features: Tensor) -> Tensor:
        return self.mlp(features.mean(dim=(2, 3))).softmax(dim=1)


class CORE(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.depthwise_separable_expert = DepthwiseSeparableExpert(channels)
        self.dilated_expert = DilatedExpert(channels)
        self.asymmetric_expert = AsymmetricExpert(channels)
        self.high_frequency_expert = HighFrequencyExpert(channels)
        self.projection = nn.Conv2d(channels, 2 * channels, 1)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, features: Tensor, routing_weights: Tensor) -> Tensor:
        responses = torch.stack((
            self.depthwise_separable_expert(features),
            self.dilated_expert(features),
            self.asymmetric_expert(features),
            self.high_frequency_expert(features),
        ), dim=1)
        fused = (routing_weights[:, :, None, None, None] * responses).sum(dim=1)
        return self.projection(fused)


class PAMoEAdapter(nn.Module):

    def __init__(self, base_qkv: nn.Linear):
        super().__init__()
        channels = base_qkv.in_features
        self.base_qkv = base_qkv
        self.core = CORE(channels)
        self.dynamic_router = DynamicRouter(channels)
        self.residual_scale = nn.Parameter(torch.tensor([1e-3]))

    def forward(self, tokens: Tensor) -> Tensor:
        q, k, v = self.base_qkv(tokens).chunk(3, dim=-1)
        features = tokens.permute(0, 3, 1, 2).contiguous()
        updates = self.core(features, self.dynamic_router(features))
        delta_q, delta_v = updates.permute(0, 2, 3, 1).contiguous().chunk(2, dim=-1)
        scale = self.residual_scale.view(1, 1, 1, 1)
        return torch.cat((q + scale * delta_q, k, v + scale * delta_v), dim=-1)
