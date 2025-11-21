import torch
from lightning_fabric.utilities.seed import seed_everything
from .files import load_image_map, ensure_csv_parent, load_image_from_file, generate_key
from .plot import plot_far_frr_with_eer, plot_single_frame


def seed_experiment(seed: int) -> None:
    seed_everything(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


__all__ = [
    "load_image_map",
    "ensure_csv_parent",
    "load_image_from_file",
    "generate_key",
    "plot_far_frr_with_eer",
    "plot_single_frame",
    "seed_experiment",
]
