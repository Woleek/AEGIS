"""Vision Transformer implementation for TransFace.

Adapted from: https://github.com/DanJun6737/TransFace
Paper: TransFace: Calibrating Transformer Training for Face Recognition (ICCV 2023)
"""

from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def drop_path(
    x: torch.Tensor, drop_prob: float = 0.0, training: bool = False
) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob: float = 0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    """MLP as used in Vision Transformer."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Multi-head self-attention mechanism."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        with torch.amp.autocast("cuda", enabled=False):
            attn = (q.float() @ k.float().transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """Transformer block with attention and MLP."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(
        self,
        img_size: int = 112,
        patch_size: int = 9,
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        # TransFace uses floor division for patch count
        num_patches = (img_size // patch_size) ** 2
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer for face recognition (TransFace architecture).

    This implementation matches the exact architecture from the TransFace paper
    and official checkpoint format. Key differences from standard ViT:
    - No cls_token - uses all patch embeddings
    - Feature head takes flattened patches (num_patches * embed_dim)
    - SE-Net style attention for patch reweighting

    Args:
        img_size: Input image size
        patch_size: Patch size
        in_chans: Number of input channels
        num_classes: Number of output classes (embedding dimension)
        embed_dim: Embedding dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dim ratio
        qkv_bias: Add bias to qkv projections
        qk_scale: Override default qk scale
        drop_rate: Dropout rate
        attn_drop_rate: Attention dropout rate
        drop_path_rate: Stochastic depth rate
        norm_layer: Normalization layer
        mask_ratio: Ratio of masked patches for training
        using_checkpoint: Use gradient checkpointing
    """

    def __init__(
        self,
        img_size: int = 112,
        patch_size: int = 9,
        in_chans: int = 3,
        num_classes: int = 512,
        embed_dim: int = 512,
        depth: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        mask_ratio: float = 0.1,
        using_checkpoint: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
        self.using_checkpoint = using_checkpoint

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches

        # TransFace: No cls_token, positional embeddings only for patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # Feature head: takes flattened patches (num_patches * embed_dim) -> 512
        flat_dim = num_patches * embed_dim
        self.feature = nn.Sequential(
            nn.Linear(flat_dim, num_classes, bias=False),
            nn.BatchNorm1d(num_classes),
            nn.Linear(num_classes, num_classes, bias=False),
            nn.BatchNorm1d(num_classes),
        )

        # SE-Net style attention: reweights patches based on their importance
        self.senet = nn.Sequential(
            nn.Linear(flat_dim, num_patches),
            nn.ReLU(inplace=True),
            nn.Linear(num_patches, num_patches),
        )

        # Weight initialization
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x: torch.Tensor, mask_ratio: float):
        """Random masking for training."""
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))

        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]

        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # Append mask tokens for masked positions
        mask_tokens = self.mask_token.expand(B, N - len_keep, -1)
        x_masked = torch.cat([x_masked, mask_tokens], dim=1)

        # Restore to original order
        x_masked = torch.gather(
            x_masked, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D)
        )
        return x_masked

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Apply random masking during training
        if self.training and self.mask_ratio > 0:
            x = self.random_masking(x, self.mask_ratio)

        for blk in self.blocks:
            if self.using_checkpoint:
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)

        x = self.norm(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)  # (B, num_patches, embed_dim)

        # Flatten all patches
        B = x.shape[0]
        x_flat = x.reshape(B, -1)  # (B, num_patches * embed_dim)

        # SE-Net attention weights
        se_weights = F.sigmoid(self.senet(x_flat))  # (B, num_patches)

        # Weight patches and flatten again
        x_weighted = x * se_weights.unsqueeze(-1)  # (B, num_patches, embed_dim)
        x_weighted_flat = x_weighted.reshape(B, -1)  # (B, num_patches * embed_dim)

        # Feature head
        out = self.feature(x_weighted_flat)
        return out


def vit_t(
    drop_path_rate: float = 0.1, mask_ratio: float = 0.1, **kwargs
) -> VisionTransformer:
    """TransFace-T (Tiny) model."""
    return VisionTransformer(
        img_size=112,
        patch_size=9,
        embed_dim=256,
        depth=12,
        num_heads=8,
        mlp_ratio=4.0,
        drop_path_rate=drop_path_rate,
        mask_ratio=mask_ratio,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def vit_s(
    drop_path_rate: float = 0.1, mask_ratio: float = 0.1, **kwargs
) -> VisionTransformer:
    """TransFace-S (Small) model."""
    return VisionTransformer(
        img_size=112,
        patch_size=9,
        embed_dim=512,
        depth=12,
        num_heads=8,
        mlp_ratio=4.0,
        drop_path_rate=drop_path_rate,
        mask_ratio=mask_ratio,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def vit_b(
    drop_path_rate: float = 0.1, mask_ratio: float = 0.1, **kwargs
) -> VisionTransformer:
    """TransFace-B (Base) model."""
    return VisionTransformer(
        img_size=112,
        patch_size=9,
        embed_dim=512,
        depth=24,
        num_heads=8,
        mlp_ratio=4.0,
        drop_path_rate=drop_path_rate,
        mask_ratio=mask_ratio,
        using_checkpoint=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def vit_l(
    drop_path_rate: float = 0.1, mask_ratio: float = 0.1, **kwargs
) -> VisionTransformer:
    """TransFace-L (Large) model."""
    return VisionTransformer(
        img_size=112,
        patch_size=9,
        embed_dim=768,
        depth=24,
        num_heads=8,
        mlp_ratio=4.0,
        drop_path_rate=drop_path_rate,
        mask_ratio=mask_ratio,
        using_checkpoint=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def get_model(name: str, **kwargs) -> VisionTransformer:
    """Factory function to get TransFace models."""
    models = {
        "vit_t": partial(vit_t, drop_path_rate=0.1, mask_ratio=0.1),
        "vit_t_dp005_mask_0": partial(vit_t, drop_path_rate=0.05, mask_ratio=0.0),
        "vit_s": partial(vit_s, drop_path_rate=0.1, mask_ratio=0.1),
        "vit_s_dp005_mask_0": partial(vit_s, drop_path_rate=0.05, mask_ratio=0.0),
        "vit_b": partial(vit_b, drop_path_rate=0.1, mask_ratio=0.1),
        "vit_b_dp005_mask_005": partial(vit_b, drop_path_rate=0.05, mask_ratio=0.05),
        "vit_l": partial(vit_l, drop_path_rate=0.1, mask_ratio=0.1),
        "vit_l_dp005_mask_005": partial(vit_l, drop_path_rate=0.05, mask_ratio=0.05),
    }
    if name not in models:
        raise ValueError(f"Unknown model: {name}. Available: {list(models.keys())}")
    return models[name](**kwargs)
