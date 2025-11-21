from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from gaussian_renderer import FlameGaussianModel, GaussianModel, render
from utils.viewer_utils import OrbitCamera


class PipelineConfig:
    debug: bool = False
    compute_cov3D_python: bool = False
    convert_SHs_python: bool = False
    antialiasing: bool = False

    def __init__(self, **kwargs):
        self.background_color = kwargs.get("background_color", [1, 1, 1])
        self.slider_scaling_modifier = kwargs.get("slider_scaling_modifier", 1.0)


def reset_flame_params(gaussians: FlameGaussianModel):
    gaussians.update_mesh_by_param_dict(
        {  # reset FLAME params
            "expr": torch.zeros(1, gaussians.n_expr),
            "rotation": torch.zeros(1, 3),
            "neck": torch.zeros(1, 3),
            "jaw": torch.zeros(1, 3),
            "eyes": torch.zeros(1, 6),
            "translation": torch.zeros(1, 3),
        }
    )


def load_gaussians(
    point_path: Path, select_regions: List[str] = []
) -> FlameGaussianModel | GaussianModel | Tuple[FlameGaussianModel, torch.Tensor]:
    """
    Load gaussians from a PLY file. If the corresponding FLAME parameters file exists,
    it loads a FlameGaussianModel; otherwise, it loads a GaussianModel.

    Args:
        point_path (Path): Path to the PLY file containing the point cloud data.
        select_regions (List[str], optional): List of regions to select. Defaults to []. Selected region can be one of:
         - left_eye,
         - right_eye,
         - eyes,
         - lips,
         - nose,
         - ears,
         - neck,
         - forehead,
         - face,
         - left_half,
         - right_half
    """
    # load gaussians
    if (Path(point_path).parent / "flame_param.npz").exists():
        gaussians = FlameGaussianModel(sh_degree=3)
    else:
        gaussians = GaussianModel(sh_degree=3)
        if select_regions:
            raise ValueError("select_regions is only supported for FlameGaussianModel.")

    # unselected_fid = gaussians.flame_model.mask.get_fid_except_fids(selected_fid)
    unselected_fid = []

    if point_path.exists():
        gaussians.load_ply(
            point_path, has_target=False, motion_path=None, disable_fid=unselected_fid
        )
        if select_regions:
            if "eyes" in select_regions:
                select_regions.remove("eyes")
                select_regions += ["left_eye", "right_eye"]
            selected_fid = gaussians.flame_model.mask.get_fid_by_region(select_regions)
            mask = ~(gaussians.binding[:, None] != selected_fid[None, :]).all(-1)
    else:
        raise FileNotFoundError(f"{point_path} does not exist.")

    reset_flame_params(gaussians)
    if select_regions:
        return gaussians, mask
    return gaussians


def prepare_camera(root_cam: OrbitCamera):
    class Cam:
        FoVx = float(np.radians(root_cam.fovx))
        FoVy = float(np.radians(root_cam.fovy))
        image_height = root_cam.image_height
        image_width = root_cam.image_width
        world_view_transform = (
            torch.tensor(root_cam.world_view_transform).float().cuda().T
        )  # the transpose is required by gaussian splatting rasterizer
        full_proj_transform = (
            torch.tensor(root_cam.full_proj_transform).float().cuda().T
        )  # the transpose is required by gaussian splatting rasterizer
        camera_center = torch.tensor(root_cam.pose[:3, 3]).cuda()

    return Cam


def render_single_frame(
    gaussians: GaussianModel | FlameGaussianModel,
    root_cam: OrbitCamera,
    pipeline: PipelineConfig,
) -> torch.Tensor:
    # Perform rendering
    cam = prepare_camera(root_cam)
    rendering = render(
        cam,
        gaussians,
        pipeline,
        torch.tensor(pipeline.background_color).type(torch.float32).cuda(),
        scaling_modifier=pipeline.slider_scaling_modifier,
    )["render"]  # (C, H, W)

    return rendering.permute(1, 2, 0)  # (H, W, C)
