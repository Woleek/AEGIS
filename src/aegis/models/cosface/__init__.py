"""CosFace model loader.

Loads an IR-SE50 Backbone from a checkpoint trained with CosFace (LMCL) loss.
"""

import torch

from .irse import Backbone


def load_cosface_model(checkpoint_path: str, device: str = "cuda") -> Backbone:
    """Load a pretrained CosFace (IR-SE50) model from a checkpoint.

    Args:
        checkpoint_path: Path to the .pth checkpoint file.
        device: Device to load the model onto ('cpu' or 'cuda').

    Returns:
        Backbone model in eval mode.
    """
    model = Backbone(num_layers=50, drop_ratio=0.6, mode="ir_se")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=torch.device(device),
        weights_only=False,
    )

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Remove DataParallel 'module.' prefix if present
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    return model


__all__ = ["load_cosface_model", "Backbone"]
