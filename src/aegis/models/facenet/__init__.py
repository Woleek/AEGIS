"""FaceNet model loader.

Loads an InceptionResnetV1 backbone from a checkpoint file.
"""

import torch

from .inception_resnet_v1 import InceptionResnetV1


def load_facenet_model(checkpoint_path: str, device: str = "cuda") -> InceptionResnetV1:
    """Load a pretrained FaceNet (InceptionResnetV1) model from a checkpoint.

    Args:
        checkpoint_path: Path to the .pth checkpoint file.
        device: Device to load the model onto ('cpu' or 'cuda').

    Returns:
        InceptionResnetV1 model in eval mode.
    """
    model = InceptionResnetV1(classify=False, num_classes=8631)

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


__all__ = ["load_facenet_model", "InceptionResnetV1"]
