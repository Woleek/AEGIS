"""Shared InsightFace IR/IR-SE and MobileFaceNet backbone architectures.

State dict key layout is preserved exactly so pretrained .pth files load
without remapping:
  IrseBackbone  — input_layer.*, body.*, output_layer.*
  MobileFaceNetBackbone — conv1.*, conv2_dw.*, conv_23.*, ..., linear.*, bn.*
"""

from __future__ import annotations

from collections import namedtuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

class _Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


# ---------------------------------------------------------------------------
# IR residual blocks
# ---------------------------------------------------------------------------

class _SEModule(nn.Module):
    def __init__(self, channels: int, reduction: int) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class _BottleneckIR(nn.Module):
    def __init__(self, in_channel: int, depth: int, stride: int) -> None:
        super().__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                nn.BatchNorm2d(depth),
            )
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            nn.BatchNorm2d(depth),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res_layer(x) + self.shortcut_layer(x)


class _BottleneckIRSE(nn.Module):
    def __init__(self, in_channel: int, depth: int, stride: int) -> None:
        super().__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                nn.BatchNorm2d(depth),
            )
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            nn.BatchNorm2d(depth),
            _SEModule(depth, 16),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res_layer(x) + self.shortcut_layer(x)


_BlockSpec = namedtuple("_BlockSpec", ["in_channel", "depth", "stride"])


def _get_block(in_channel: int, depth: int, num_units: int, stride: int = 2) -> list[_BlockSpec]:
    return [_BlockSpec(in_channel, depth, stride)] + [
        _BlockSpec(depth, depth, 1) for _ in range(num_units - 1)
    ]


def _get_blocks(num_layers: int) -> list[list[_BlockSpec]]:
    if num_layers == 50:
        return [
            _get_block(64, 64, 3),
            _get_block(64, 128, 4),
            _get_block(128, 256, 14),
            _get_block(256, 512, 3),
        ]
    if num_layers == 100:
        return [
            _get_block(64, 64, 3),
            _get_block(64, 128, 13),
            _get_block(128, 256, 30),
            _get_block(256, 512, 3),
        ]
    if num_layers == 152:
        return [
            _get_block(64, 64, 3),
            _get_block(64, 128, 8),
            _get_block(128, 256, 36),
            _get_block(256, 512, 3),
        ]
    raise ValueError(f"num_layers must be 50, 100, or 152; got {num_layers}")


# ---------------------------------------------------------------------------
# IR / IR-SE backbone
# ---------------------------------------------------------------------------

class IrseBackbone(nn.Module):
    """IR or IR-SE face recognition backbone for 112×112 inputs.

    Produces raw 512-D BN features; the embedder wrapper L2-normalizes.

    Derived from TreB1eN/InsightFace_Pytorch:
        input_layer.{0,1,2}.*   Conv-BN-PReLU stem
        body.N.*                 bottleneck stack
        output_layer.{0,3,4}.*  BN-Dropout-Flatten-Linear-BN head
    """

    def __init__(self, num_layers: int, drop_ratio: float, mode: str) -> None:
        super().__init__()
        assert num_layers in (50, 100, 152), f"num_layers must be 50, 100, or 152; got {num_layers}"
        assert mode in ("ir", "ir_se"), f"mode must be 'ir' or 'ir_se'; got {mode}"

        unit_module = _BottleneckIR if mode == "ir" else _BottleneckIRSE
        blocks = _get_blocks(num_layers)

        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
        )
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(drop_ratio),
            _Flatten(),
            nn.Linear(512 * 7 * 7, 512),
            nn.BatchNorm1d(512),
        )
        modules: list[nn.Module] = []
        for block in blocks:
            for b in block:
                modules.append(unit_module(b.in_channel, b.depth, b.stride))
        self.body = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return x


# ---------------------------------------------------------------------------
# MobileFaceNet backbone
# ---------------------------------------------------------------------------

class _ConvBlock(nn.Module):
    def __init__(
        self,
        in_c: int,
        out_c: int,
        kernel: tuple[int, int] = (1, 1),
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (0, 0),
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_c, out_c, kernel_size=kernel, groups=groups,
            stride=stride, padding=padding, bias=False,
        )
        self.bn = nn.BatchNorm2d(out_c)
        self.prelu = nn.PReLU(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.prelu(self.bn(self.conv(x)))


class _LinearBlock(nn.Module):
    def __init__(
        self,
        in_c: int,
        out_c: int,
        kernel: tuple[int, int] = (1, 1),
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (0, 0),
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_c, out_c, kernel_size=kernel, groups=groups,
            stride=stride, padding=padding, bias=False,
        )
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class _DepthWise(nn.Module):
    def __init__(
        self,
        in_c: int,
        out_c: int,
        residual: bool = False,
        kernel: tuple[int, int] = (3, 3),
        stride: tuple[int, int] = (2, 2),
        padding: tuple[int, int] = (1, 1),
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.conv = _ConvBlock(in_c, groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = _ConvBlock(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride)
        self.project = _LinearBlock(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        return short_cut + x if self.residual else x


class _Residual(nn.Module):
    def __init__(
        self,
        c: int,
        num_block: int,
        groups: int,
        kernel: tuple[int, int] = (3, 3),
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (1, 1),
    ) -> None:
        super().__init__()
        self.model = nn.Sequential(*[
            _DepthWise(c, c, residual=True, kernel=kernel, padding=padding, stride=stride, groups=groups)
            for _ in range(num_block)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MobileFaceNetBackbone(nn.Module):
    """MobileFaceNet backbone for 112×112 inputs → 512-D embedding.

    Produces raw BN features; the embedder wrapper L2-normalizes.
    """

    def __init__(self, embedding_size: int = 512) -> None:
        super().__init__()
        self.conv1 = _ConvBlock(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = _ConvBlock(64, 64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64)
        self.conv_23 = _DepthWise(64, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        self.conv_3 = _Residual(64, num_block=4, groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_34 = _DepthWise(64, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        self.conv_4 = _Residual(128, num_block=6, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_45 = _DepthWise(128, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512)
        self.conv_5 = _Residual(128, num_block=2, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_6_sep = _ConvBlock(128, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_6_dw = _LinearBlock(512, 512, groups=512, kernel=(7, 7), stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = _Flatten()
        self.linear = nn.Linear(512, embedding_size, bias=False)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2_dw(out)
        out = self.conv_23(out)
        out = self.conv_3(out)
        out = self.conv_34(out)
        out = self.conv_4(out)
        out = self.conv_45(out)
        out = self.conv_5(out)
        out = self.conv_6_sep(out)
        out = self.conv_6_dw(out)
        out = self.conv_6_flatten(out)
        out = self.linear(out)
        out = self.bn(out)
        return out
