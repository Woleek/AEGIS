"""SwinFace model implementation for face recognition.

Based on: https://github.com/lxq1000/SwinFace
"""

import torch
import torch.nn as nn
from .swin import SwinTransformer
from .subnets import FeatureAttentionModule, TaskSpecificSubnets, OutputModule, ModelBox


class SwinFaceCfg:
    """Configuration for SwinFace model (Swin-T backbone)."""

    network = "swin_t"
    fam_kernel_size = 3
    fam_in_chans = 2112
    fam_conv_shared = False
    fam_conv_mode = "split"
    fam_channel_attention = "CBAM"
    fam_spatial_attention = None
    fam_pooling = "max"
    fam_la_num_list = [2 for _ in range(11)]
    fam_feature = "all"
    fam = "3x3_2112_F_s_C_N_max"
    embedding_size = 512


def build_model(cfg: SwinFaceCfg) -> ModelBox:
    """Build the SwinFace model from configuration."""
    backbone = SwinTransformer(num_classes=cfg.embedding_size)

    fam = FeatureAttentionModule(
        in_chans=cfg.fam_in_chans,
        kernel_size=cfg.fam_kernel_size,
        conv_shared=cfg.fam_conv_shared,
        conv_mode=cfg.fam_conv_mode,
        channel_attention=cfg.fam_channel_attention,
        spatial_attention=cfg.fam_spatial_attention,
        pooling=cfg.fam_pooling,
        la_num_list=cfg.fam_la_num_list,
    )
    tss = TaskSpecificSubnets()
    om = OutputModule()

    model = ModelBox(
        backbone=backbone, fam=fam, tss=tss, om=om, feature=cfg.fam_feature
    )
    return model


def load_swinface_model(checkpoint_path: str, device: str = "cuda") -> ModelBox:
    """Load SwinFace model with pretrained weights.

    Args:
        checkpoint_path: Path to the checkpoint file (.pt)
        device: Device to load the model on

    Returns:
        Loaded ModelBox with pretrained weights
    """
    cfg = SwinFaceCfg()
    model = build_model(cfg)

    checkpoint = torch.load(
        checkpoint_path, map_location=torch.device(device), weights_only=False
    )

    model.backbone.load_state_dict(checkpoint["state_dict_backbone"])
    model.fam.load_state_dict(checkpoint["state_dict_fam"])
    model.tss.load_state_dict(checkpoint["state_dict_tss"])
    model.om.load_state_dict(checkpoint["state_dict_om"])

    model.eval()
    return model


__all__ = [
    "SwinFaceCfg",
    "build_model",
    "load_swinface_model",
    "ModelBox",
    "SwinTransformer",
]
