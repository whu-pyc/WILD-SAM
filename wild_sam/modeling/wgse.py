from typing import Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class HaarDWT(nn.Module):

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        if x.shape[-2] % 2 or x.shape[-1] % 2:
            raise ValueError("HaarDWT requires even spatial dimensions")
        even_rows = x[:, :, 0::2, :] / 2
        odd_rows = x[:, :, 1::2, :] / 2
        tl, bl = even_rows[:, :, :, 0::2], odd_rows[:, :, :, 0::2]
        tr, br = even_rows[:, :, :, 1::2], odd_rows[:, :, :, 1::2]
        ll = tl + bl + tr + br
        lh = -tl + bl - tr + br
        hl = -tl - bl + tr + br
        hh = tl - bl - tr + br
        return ll, lh, hl, hh


def _tokens(feature: Tensor) -> Tensor:
    return feature.flatten(2).transpose(1, 2).contiguous()


def _heads(tokens: Tensor, num_heads: int) -> Tensor:
    batch, length, channels = tokens.shape
    return tokens.reshape(batch, length, num_heads, channels // num_heads).transpose(1, 2)


def _spatial_attention(q: Tensor, k: Tensor, v: Tensor, num_heads: int) -> Tensor:
    batch, length, channels = q.shape
    scale = (channels // num_heads) ** -0.5
    query = _heads(q, num_heads)
    key = _heads(k, num_heads)
    value = _heads(v, num_heads)
    weights = ((query * scale) @ key.transpose(-2, -1)).softmax(dim=-1)
    return (weights @ value).transpose(1, 2).contiguous().view(batch, length, channels)


class AFM(nn.Module):

    def __init__(self, channels: int, num_heads: int):
        super().__init__()
        if num_heads < 1 or channels % num_heads:
            raise ValueError("channels must be divisible by a positive num_heads")
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, 3 * channels)
        self.attention_projection = nn.Linear(channels, channels)
        self.attention_scale = nn.Parameter(torch.tensor([0.1], dtype=torch.float32))
        self.depthwise_conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.pointwise_conv = nn.Conv2d(channels, channels, 1, bias=False)
        self.conv_norm = nn.BatchNorm2d(channels)
        self.conv_scale = nn.Parameter(torch.tensor([0.1], dtype=torch.float32))

    def forward(self, feature: Tensor) -> Tensor:
        query, key, value = self.qkv(self.norm(_tokens(feature))).chunk(3, dim=-1)
        attended = _spatial_attention(query, key, value, self.num_heads)
        attended = self.attention_projection(attended).transpose(1, 2).reshape_as(feature).contiguous()
        feature = feature + self.attention_scale.view(1, 1, 1, 1) * attended
        local = self.pointwise_conv(self.depthwise_conv(feature))
        local = F.gelu(self.conv_norm(local))
        return feature + self.conv_scale.view(1, 1, 1, 1) * local


class SubbandCrossAttention(nn.Module):

    def __init__(self, channels: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.query_norm = nn.LayerNorm(channels)
        self.context_norm = nn.LayerNorm(channels)
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        self.projection = nn.Linear(channels, channels)
        self.residual_scale = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, feature: Tensor, context: Tensor) -> Tensor:
        query = self.query(self.query_norm(_tokens(feature)))
        context_tokens = self.context_norm(_tokens(context))
        attended = _spatial_attention(query, self.key(context_tokens), self.value(context_tokens), self.num_heads)
        update = self.projection(attended).transpose(1, 2).reshape_as(feature).contiguous()
        return feature + self.residual_scale.view(1, 1, 1, 1) * update


class SCB(nn.Module):

    def __init__(self, channels: int, num_heads: int):
        super().__init__()
        self.lh = SubbandCrossAttention(channels, num_heads)
        self.hl = SubbandCrossAttention(channels, num_heads)
        self.hh = SubbandCrossAttention(channels, num_heads)

    def forward(self, lh: Tensor, hl: Tensor, hh: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        return self.lh(lh, (hl + hh) / 2), self.hl(hl, (lh + hh) / 2), self.hh(hh, (lh + hl) / 2)


class WDFR(nn.Module):

    def __init__(self, channels: int, num_heads: int):
        super().__init__()
        self.afm_lh = AFM(channels, num_heads)
        self.afm_hl = AFM(channels, num_heads)
        self.afm_hh = AFM(channels, num_heads)
        self.scb = SCB(channels, num_heads)

    def forward(self, lh: Tensor, hl: Tensor, hh: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        return self.scb(self.afm_lh(lh), self.afm_hl(hl), self.afm_hh(hh))


class SEGate(nn.Module):

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(16, channels // reduction)
        self.reduce = nn.Conv2d(channels, hidden, 1)
        self.expand = nn.Conv2d(hidden, channels, 1)

    def forward(self, feature: Tensor) -> Tensor:
        descriptor = feature.mean(dim=(2, 3), keepdim=True)
        weights = torch.sigmoid(self.expand(F.gelu(self.reduce(descriptor))))
        return feature * weights


class WGSE(nn.Module):

    def __init__(self, channels: int = 256, num_heads: int = 8):
        super().__init__()
        self.dwt = HaarDWT()
        self.wdfr = WDFR(channels, num_heads)
        self.fusion_projection = nn.Conv2d(3 * channels, 2 * channels, 1, bias=False)
        self.fusion_norm = nn.BatchNorm2d(2 * channels)
        self.se_gate = SEGate(channels)
        self.fusion_scale = nn.Parameter(torch.tensor([0.1], dtype=torch.float32))
        self.prompt_scale = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, embedding: Tensor) -> Tensor:
        _, lh, hl, hh = self.dwt(embedding)
        lh, hl, hh = self.wdfr(lh, hl, hh)
        combined = self.fusion_projection(torch.cat((lh, hl, hh), dim=1))
        values, gates = F.gelu(self.fusion_norm(combined)).chunk(2, dim=1)
        gated = self.se_gate(values * torch.sigmoid(gates))
        mean = (lh + hl + hh) / 3
        high_frequency_prompt = mean + self.fusion_scale.view(1, 1, 1, 1) * (gated - mean)
        prompt = F.interpolate(high_frequency_prompt, size=embedding.shape[-2:], mode="bilinear", align_corners=False)
        return embedding + self.prompt_scale.view(1, 1, 1, 1) * prompt
