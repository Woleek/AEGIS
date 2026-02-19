import sys
from ..config import ROOT_DIR

sys.path.append((ROOT_DIR.resolve() / "GaussianAvatars").as_posix())

from pathlib import Path
from typing import Optional, Tuple
import foolbox as fb
from plyfile import PlyData

from gaussian_renderer import FlameGaussianModel
import torch
from tqdm import tqdm


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


# Available regions:
# Eyes: left_eye, right_eye, left_eyeball, right_eyeball, left_eye_region, right_eye_region, eyeballs, left_eyelid, right_eyelid, eyelids, left_iris, right_iris, irises, sclerae, eye_region
# Mouth: lips, lips_tight, lip_inside, lip_inside_ring, lip_inside_upper, lip_inside_lower
# Face: face, nose, forehead, ears, left_ear, right_ear, skin, hair, scalp
# Neck: neck, neck_top, neck_upper, neck_lower, neck_base
# Other: boundary, bottomline
DEFAULT_REGION_MULTIPLIERS = {
    # Identity-critical regions: higher epsilon for stronger privacy protection
    "left_eye": 1.3,  # Eye region + eyeball (includes left_eye_region + left_eyeball)
    "right_eye": 1.3,  # Eye region + eyeball (includes right_eye_region + right_eyeball)
    "nose": 1.2,  # Nose bridge and tip
    # Facial features: balanced epsilon
    "lips": 1.0,  # Mouth region
    "forehead": 0.6,  # Upper face
    # Less critical regions: lower epsilon to reduce visible artifacts
    "skin": 0.7,  # General skin (auto-computed, excludes other regions)
    "ears": 0.7,  # Both ears combined
    "neck": 0.5,  # Neck region
    "scalp": 0.4,  # Hair/scalp region
}


def create_adaptive_epsilon_mask(
    gaussians: FlameGaussianModel,
    base_epsilon: float,
    region_multipliers: Optional[dict[str, float]] = None,
) -> torch.Tensor:
    """
    Create per-Gaussian epsilon values based on facial region.

    Args:
        gaussians: FlameGaussianModel with binding to FLAME faces
        base_epsilon: Base epsilon value to scale
        region_multipliers: Dict mapping region names to multipliers.
            Uses DEFAULT_REGION_MULTIPLIERS if not provided.

    Returns:
        Tensor of shape (N,) with per-Gaussian epsilon values
    """
    if region_multipliers is None:
        region_multipliers = DEFAULT_REGION_MULTIPLIERS.copy()

    n_gaussians = gaussians._xyz.shape[0]
    device = gaussians._xyz.device

    # Start with base epsilon for all Gaussians
    epsilon_per_gaussian = torch.ones(n_gaussians, device=device) * base_epsilon

    # Get binding from Gaussians to FLAME faces
    binding = gaussians.binding  # (N,) face indices

    # Apply region-specific multipliers
    for region, multiplier in region_multipliers.items():
        try:
            region_fids = gaussians.flame_model.mask.get_fid_by_region([region])
            if len(region_fids) == 0:
                continue
            # Convert to set
            region_fids_set = set(region_fids.cpu().numpy().tolist())
            # Create mask
            region_mask = torch.tensor(
                [b.item() in region_fids_set for b in binding.cpu()],
                dtype=torch.bool,
                device=device,
            )
            epsilon_per_gaussian[region_mask] = base_epsilon * multiplier
        except Exception:
            # Region may not exist in this FLAME model
            continue

    return epsilon_per_gaussian


def adaptive_linf_pgd_attack(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    epsilon_per_element: torch.Tensor,
    target_class: torch.Tensor,
    steps: int = 300,
    step_size: Optional[float] = None,
    random_start: bool = False,
) -> torch.Tensor:
    """
    L-infinity PGD attack with per-element epsilon bounds.

    This implementation supports different epsilon values for each element,
    enabling region-adaptive perturbation budgets.

    Args:
        model: Differentiable model that outputs logits
        inputs: Input tensor of shape (N, 1, C) or (N, C)
        epsilon_per_element: Per-element epsilon of shape (N,)
        target_class: Target class for targeted attack
        steps: Number of PGD steps
        step_size: Step size per iteration (default: 2.5 * eps / steps)
        random_start: Whether to start from random perturbation

    Returns:
        Adversarial tensor with same shape as inputs
    """
    device = inputs.device
    inputs = inputs.detach()
    original_shape = inputs.shape

    # Flatten to (N, C) if needed
    if inputs.dim() == 3 and inputs.shape[1] == 1:
        inputs = inputs.squeeze(1)  # (N, 1, C) -> (N, C)

    # Epsilon shape: (N, 1) for proper broadcasting with (N, C) inputs
    eps = epsilon_per_element.view(-1, 1).to(device)

    # Default step size
    if step_size is None:
        step_size = 2.5 * eps.mean().item() / steps

    # Initialize delta (perturbation)
    if random_start:
        delta = torch.empty_like(inputs).uniform_(-1, 1)
        delta = delta * eps  # Scale by per-element epsilon
    else:
        delta = torch.zeros_like(inputs)

    delta.requires_grad_(True)

    for _ in tqdm(range(steps), desc="PGD Attack Steps", leave=False):
        if delta.grad is not None:
            delta.grad.zero_()

        # Forward pass with perturbed input
        x_adv = inputs + delta
        # Reshape back if model expects (N, 1, C)
        if len(original_shape) == 3:
            x_adv_model = x_adv.unsqueeze(1)
        else:
            x_adv_model = x_adv
        logits = model(x_adv_model)

        # Untargeted attack: minimize logit of original class (target_class)
        # This causes misclassification away from the original class
        loss = logits[0, target_class[0]]

        # Backward pass
        loss.backward()

        # PGD step: gradient sign * step_size
        with torch.no_grad():
            grad_sign = delta.grad.sign()
            delta_new = delta - step_size * grad_sign

            # Project back to epsilon ball (per-element)
            delta_new = torch.where(delta_new > eps, eps, delta_new)
            delta_new = torch.where(delta_new < -eps, -eps, delta_new)

        delta = delta_new.clone().detach().requires_grad_(True)

    result = inputs + delta.detach()

    # Restore original shape
    if len(original_shape) == 3:
        result = result.unsqueeze(1)

    return result


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
