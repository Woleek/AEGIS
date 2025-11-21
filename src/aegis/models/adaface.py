import os
from collections import namedtuple
from pathlib import Path
from typing import Dict, List, Literal, Sequence
import cv2
import insightface
from insightface.utils.face_align import estimate_norm
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from torchvision import transforms

from .base import chunk_sequence, warp_affine_pytorch
from .base import (
    BaseEmbedder,
    load_insightface_detector,
    resolve_compute_device,
)
from tqdm import tqdm


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class BasicBlockIR(nn.Module):
    """BasicBlock for IRNet"""

    def __init__(self, in_channel, depth, stride):
        super(BasicBlockIR, self).__init__()
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
            nn.BatchNorm2d(depth),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            nn.BatchNorm2d(depth),
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut


def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + [
        Bottleneck(depth, depth, 1) for _ in range(num_units - 1)
    ]


class Bottleneck(namedtuple("Block", ["in_channel", "depth", "stride"])):
    """A named tuple describing a ResNet block."""


def get_blocks(num_layers=100):
    if num_layers == 50:
        return [
            get_block(64, 64, 3),
            get_block(64, 128, 4),
            get_block(128, 256, 14),
            get_block(256, 512, 3),
        ]
    elif num_layers == 100:
        return [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3),
        ]
    else:
        raise ValueError(
            "num_layers should be 50 or 100, but got {}".format(num_layers)
        )


def initialize_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            if m.weight is not None:
                m.weight.data.fill_(1)
            if m.bias is not None:
                m.bias.data.zero_()


class Backbone(nn.Module):
    def __init__(self, input_size, num_layers, mode="ir"):
        """Args:
        input_size: input_size of backbone
        num_layers: num_layers of backbone
        mode: support ir or irse
        """
        super(Backbone, self).__init__()
        assert input_size[0] in [
            112,
            224,
        ], "input_size should be [112, 112] or [224, 224]"
        assert num_layers in [
            18,
            34,
            50,
            100,
            152,
            200,
        ], "num_layers should be 18, 34, 50, 100 or 152"
        assert mode in ["ir", "ir_se"], "mode should be ir or ir_se"
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False), nn.BatchNorm2d(64), nn.PReLU(64)
        )
        blocks = get_blocks(num_layers)
        unit_module = BasicBlockIR
        output_channel = 512

        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(output_channel),
            nn.Dropout(0.4),
            Flatten(),
            nn.Linear(output_channel * 7 * 7, 512),
            nn.BatchNorm1d(512, affine=False),
        )
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(
                        bottleneck.in_channel, bottleneck.depth, bottleneck.stride
                    )
                )
        self.body = nn.Sequential(*modules)

        initialize_weights(self.modules())

    def forward(self, x):
        # current code only supports one extra image
        # it comes with a extra dimension for number of extra image. We will just squeeze it out for now
        x = self.input_layer(x)

        for idx, module in enumerate(self.body):
            x = module(x)

        x = self.output_layer(x)
        norm = torch.norm(x, 2, 1, True)
        output = torch.div(x, norm)

        return output, norm


def IR_101(input_size=(112, 112)):
    return Backbone(input_size, 100, "ir")


def IR_50(input_size=(112, 112)):
    return Backbone(input_size, 50, "ir")


def load_pretrained_model(path, model_type: str = "ir101", device: str = "cuda"):
    # load model and pretrained statedict
    if model_type == "ir50":
        model = IR_50()
        statedict = torch.load(
            os.path.join(path, "adaface_ir50_ms1mv2.ckpt"),
            weights_only=False,
            map_location=torch.device(device),
        )["state_dict"]
    elif model_type == "ir101":
        model = IR_101()
        statedict = torch.load(
            os.path.join(path, "adaface_ir101_ms1mv3.ckpt"),
            weights_only=False,
            map_location=torch.device(device),
        )["state_dict"]
    model_statedict = {
        key[6:]: val for key, val in statedict.items() if key.startswith("model.")
    }
    model.load_state_dict(model_statedict)
    model.eval()
    return model


class AdaFaceTorchModel(nn.Module):
    def __init__(
        self, path: str, freeze=True, model_type: str = "ir101", device: str = "cuda"
    ):
        super(AdaFaceTorchModel, self).__init__()
        self._prepare_model(path, freeze, model_type, device)

    def _prepare_model(self, path, freeze, model_type, device):
        self.model = load_pretrained_model(path, model_type, device)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

    def forward(self, images):
        """
        Expects images as torch.Tensor with shape [B, 3, H, W] and pixel values in [0, 1]
        """
        if not torch.is_tensor(images):
            raise ValueError("Input must be a PyTorch tensor.")
        embeddings, _ = self.model(images)
        return embeddings

    def preprocess_face(self, face_crop):
        face = face_crop.float() / 255.0
        face = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(face)
        return face

    def compute_similarities(self, e_i, e_j):
        return e_i @ e_j.T


class AdaFaceEmbedder(BaseEmbedder):
    """Adapter around the repo-internal AdaFace implementation."""

    def __init__(
        self,
        device: Literal["cpu", "cuda"],
        batch_size: int,
        model_path: Path,
        model_type: Literal["ir50", "ir101"],
    ) -> None:
        self.requested_device = device
        self.device_str = resolve_compute_device(device)
        self.device = torch.device(self.device_str)
        self.batch_size = batch_size
        self.ctx_id = 0 if self.device_str == "cuda" else -1
        ort.set_default_logger_severity(3)
        self.detect_model = load_insightface_detector(self.ctx_id)
        self.model = AdaFaceTorchModel(
            path=str(model_path), model_type=model_type, device=self.device_str
        )
        self.model = self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _norm(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

    def _align_face(self, image: np.ndarray) -> np.ndarray:
        try:
            bboxes, kpss = self.detect_model.detect(image, max_num=1)
            if kpss is not None and len(kpss) > 0:
                return insightface.utils.face_align.norm_crop(image, landmark=kpss[0])
        except Exception as e:
            print(f"Error aligning face: {e}")
            pass
        return cv2.resize(image, (112, 112), interpolation=cv2.INTER_LINEAR)

    def _prepare_tensor(self, aligned_image: np.ndarray) -> torch.Tensor:
        rgb = aligned_image.copy()
        tensor = torch.from_numpy(rgb).permute(2, 0, 1)
        tensor = self.model.preprocess_face(tensor)
        return tensor

    def embed(self, paths: Sequence[Path], key_fn) -> Dict[str, np.ndarray]:
        embeddings: Dict[str, np.ndarray] = {}
        for chunk in tqdm(
            chunk_sequence(paths, self.batch_size),
            desc="Embedding images with AdaFace",
            total=len(paths) // self.batch_size + 1,
        ):
            batch_tensors: List[torch.Tensor] = []
            valid_keys: List[str] = []
            for path in chunk:
                image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
                if image is None:
                    continue
                aligned = self._align_face(image)
                tensor = self._prepare_tensor(aligned)
                batch_tensors.append(tensor)
                valid_keys.append(key_fn(path))
            if not batch_tensors:
                continue
            batch = torch.stack(batch_tensors, dim=0).to(self.device)
            with torch.no_grad():
                features = self.model(batch).detach().cpu().numpy()
            for key, feat in zip(valid_keys, features):
                embeddings[key] = self._norm(feat)
        return embeddings


class AdaFace(AdaFaceEmbedder):
    def _align_face(self, image: torch.Tensor) -> torch.Tensor:
        img_arr = image.detach().cpu().numpy() * 255
        bboxes, kpss = self.detect_model.detect(img_arr, max_num=1)
        if kpss is not None and len(kpss) > 0:
            M = estimate_norm(kpss[0], image_size=112)
            crop = warp_affine_pytorch(
                image_tensor=image.permute(2, 0, 1).unsqueeze(0).float(),
                m_matrix=torch.from_numpy(M).unsqueeze(0).to(image.device).float(),
                out_size=(112, 112),
            ).squeeze(0)
        else:
            raise ValueError("No face detected in the image for AdaFace embedding.")
        return crop

    def _prepare_tensor(self, aligned_image: torch.Tensor) -> torch.Tensor:
        tensor = aligned_image.float() * 255.0  # Scale to [0, 255]
        tensor = self.model.preprocess_face(tensor).unsqueeze(0)  # Add batch dimension
        return tensor.to(self.device)

    @staticmethod
    def _norm(vec: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(vec, p=2, dim=1, keepdim=True)
        if norm == 0:
            return vec
        return vec / norm

    # AdaFace expects (N, C, H, W) in [-1, 1]
    def embed(self, image: torch.Tensor) -> torch.Tensor:
        # cv2.imwrite("input_adaface.png", cv2.cvtColor(image.detach().cpu().numpy() * 255, cv2.COLOR_RGB2BGR).astype(np.uint8))
        # input("Check input image...")
        aligned = self._align_face(image)
        # cv2.imwrite("aligned_adaface.png", cv2.cvtColor(aligned.permute(1, 2, 0).detach().cpu().numpy() * 255, cv2.COLOR_RGB2BGR).astype(np.uint8))
        # input("Check aligned image...")
        tensor = self._prepare_tensor(aligned)
        emb = self.model(tensor)
        emb = self._norm(emb)
        return emb

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return self.embed(image)
