import sys
from ..config import ROOT_DIR

sys.path.append((ROOT_DIR.resolve() / "GaussianAvatars").as_posix())

from pathlib import Path
from typing import Optional, Tuple
import foolbox as fb
from plyfile import PlyData

from gaussian_renderer import FlameGaussianModel
import torch


def get_foolbox_attack(
    adv_attack_name: str, steps: int, random_start: bool
) -> fb.attacks.Attack:
    match adv_attack_name.lower():
        case "linfpgd":
            return fb.attacks.LinfProjectedGradientDescentAttack(
                steps=steps, random_start=random_start
            )
        case "l2pgd":
            return fb.attacks.L2ProjectedGradientDescentAttack(
                steps=steps, random_start=random_start
            )
        case "linffgsm":
            return fb.attacks.LinfFastGradientAttack(random_start=random_start)
        case "l2fgsm":
            return fb.attacks.L2FastGradientAttack(random_start=random_start)
        case "ddn":
            return fb.attacks.DDNAttack(steps=steps, init_epsilon=100)
        case _:
            raise ValueError(f"Unsupported attack name: {adv_attack_name}")


def set_targeted_features(
    gaussians: FlameGaussianModel, features_type: str, new_values: torch.Tensor
) -> None:
    match features_type:
        case "DC":
            gaussians._features_dc = new_values
        case "AC":
            gaussians._features_rest = new_values
        case "pos":
            gaussians._xyz = new_values
        case "scale":
            gaussians._scaling = new_values
        case "rot":
            gaussians._rotation = new_values
        case "opacity":
            gaussians._opacity = new_values
        case _:
            raise ValueError(f"Unsupported features type: {features_type}")


def get_targeted_features(
    gaussians: FlameGaussianModel, features_type: str
) -> torch.Tensor:
    match features_type:
        case "DC":
            return gaussians._features_dc
        case "AC":
            return gaussians._features_rest
        case "pos":
            return gaussians._xyz
        case "scale":
            return gaussians._scaling
        case "rot":
            return gaussians._rotation
        case "opacity":
            return gaussians._opacity
        case _:
            raise ValueError(f"Unsupported features type: {features_type}")


def normalize_camera_angles(
    angle_values: Optional[list[float]],
) -> list[Tuple[float, float, float]]:
    if not angle_values:
        return [(0.0, 0.0, 0.0)]

    if len(angle_values) != 6:
        raise ValueError(
            "camera_boundary_angles must contain exactly 6 float values: "
            "orbit_x_min orbit_x_max orbit_y_min orbit_y_max orbit_z_min orbit_z_max"
        )

    angles: list[Tuple[float, float, float]] = []
    for _ in range(0, len(angle_values), 3):
        orbit_x_min, orbit_x_max, orbit_y_min, orbit_y_max, orbit_z_min, orbit_z_max = (
            angle_values
        )
        angles.append(
            (
                float(orbit_x_min),
                float(orbit_x_max),
                float(orbit_y_min),
                float(orbit_y_max),
                float(orbit_z_min),
                float(orbit_z_max),
            )
        )
    return angles


def ensure_output_structure(
    output_dir: str, epsilons: list[float], avatar_id: str
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for eps in epsilons:
        eps_dir = output_path / f"eps_{eps:.3f}"
        eps_dir.mkdir(parents=True, exist_ok=True)
        (eps_dir / "renders").mkdir(parents=True, exist_ok=True)
        (eps_dir / "avatars" / avatar_id).mkdir(parents=True, exist_ok=True)


def write_ply_with_dc_colors(
    original_ply_path: Path, new_colors: torch.Tensor, output_ply_path: Path
) -> None:
    # Read the original PLY file
    try:
        plydata = PlyData.read(original_ply_path)
    except Exception as e:
        print(f"Error reading PLY file {original_ply_path}: {e}")
        return

    # Check if 'vertex' element exists
    if "vertex" not in plydata:
        raise ValueError("PLY file does not contain a 'vertex' element")

    # Get a direct reference to the vertex data
    vertices = plydata["vertex"].data

    # Prepare new colors
    if isinstance(new_colors, torch.Tensor):
        new_colors = new_colors.detach().cpu().numpy()

    # Reshape to (N, 3)
    new_colors = new_colors.squeeze()  # Remove singleton dimensions
    if new_colors.ndim == 3:
        new_colors = new_colors.reshape(-1, 3)  # Ensure it's (N, 3)

    assert new_colors.shape[0] == len(vertices), (
        f"Number of colors ({new_colors.shape[0]}) must match number of vertices ({len(vertices)})"
    )
    assert new_colors.shape[1] == 3, "DC colors must have 3 channels (RGB)"

    # Check if DC properties exist before trying to write to them
    prop_names = vertices.dtype.names
    dc_props = ["f_dc_0", "f_dc_1", "f_dc_2"]
    if not all(p in prop_names for p in dc_props):
        raise ValueError(
            f"PLY 'vertex' element is missing one or more DC properties: {dc_props}"
        )

    # Update DC color properties
    # This modifies the structured array 'vertices' in place.
    vertices["f_dc_0"] = new_colors[:, 0].astype(vertices["f_dc_0"].dtype)
    vertices["f_dc_1"] = new_colors[:, 1].astype(vertices["f_dc_1"].dtype)
    vertices["f_dc_2"] = new_colors[:, 2].astype(vertices["f_dc_2"].dtype)

    # Write to output path
    output_dir = output_ply_path.parent
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Write the modified plydata object
    plydata.write(output_ply_path)
