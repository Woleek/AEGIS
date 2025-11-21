from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Literal, Optional

from ..config import DATASETS_DIR


class DatasetIdentityLookup:
    """Abstract base class for dataset specific identity resolution."""

    def lookup(self, file_key: str) -> str:
        raise NotImplementedError


@dataclass(slots=True)
class DatasetSpec:
    """Small container describing where images and metadata live."""

    name: Literal["CelebA", "lfw", "NeRSembleGT"]
    root: Path
    images_subdir: Path
    file_extension: str
    identity_lookup: "DatasetIdentityLookup"
    celeba_test_set_only: bool = False

    @property
    def images_root(self) -> Path:
        return self.root / self.images_subdir

    @property
    def cache_dir(self) -> Path:
        return self.root / ".cache"


class CelebAIdentityLookup(DatasetIdentityLookup):
    """Maps file names (``000001.jpg``) to celebrity IDs."""

    def __init__(self, identity_file: Path, test_set_only: bool = False) -> None:
        if not identity_file.exists():
            raise FileNotFoundError(identity_file)
        self._mapping: Dict[str, str] = {}
        cutoff = 182638 if test_set_only else None
        with identity_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                file_name, identity = line.strip().split()
                if cutoff is not None:
                    index = int(Path(file_name).stem)
                    if index < cutoff:
                        continue
                self._mapping[file_name] = identity

    def lookup(self, file_key: str) -> str:
        if "___" in file_key:
            _, leaf = file_key.split("___", maxsplit=1)
            file_name = f"{leaf}.jpg"
        else:
            file_name = Path(file_key).name
        try:
            return self._mapping[file_name]
        except KeyError as exc:
            raise KeyError(f"Missing CelebA identity for {file_key}") from exc


class LFWIdentityLookup(DatasetIdentityLookup):
    """LFW identities are encoded by the parent directory name."""

    def lookup(self, file_key: str) -> str:
        if "___" in file_key:
            _, path_part = file_key.split("___", maxsplit=1)
            return Path(path_part).parent.stem
        return Path(file_key).parent.stem


class NeRSembleIdentityLookup(DatasetIdentityLookup):
    """Identity lookup for NeRSemble Gaussian Avatar assets (parent-based or stem-based)."""

    def lookup(self, file_key: str) -> str:
        if "___" in file_key:
            _, path_part = file_key.split("___", maxsplit=1)
            if Path(path_part).parent.stem:
                return Path(path_part).parent.stem
            return Path(path_part).stem
        if Path(file_key).parent.stem:
            return Path(file_key).parent.stem
        return Path(file_key).stem


class FaceDataset:
    """Enumerates face image paths under a root directory."""

    def __init__(
        self,
        root: Path,
        file_extension: str,
        celeba_test_set_only: bool = False,
    ) -> None:
        if not root.exists():
            raise FileNotFoundError(root)
        self.root = root
        self.file_extension = file_extension
        self.celeba_test_set_only = celeba_test_set_only
        self.paths: List[Path] = []
        self._discover()

    def _discover(self) -> None:
        cutoff = 182638 if self.celeba_test_set_only else None
        for path in self.root.rglob(f"*{self.file_extension}"):
            if cutoff is not None:
                try:
                    index = int(path.stem)
                except ValueError:
                    pass
                else:
                    if index < cutoff:
                        continue
            self.paths.append(path)
        self.paths.sort()

    def __len__(self) -> int:
        return len(self.paths)

    def iter_paths(self) -> Iterator[Path]:
        yield from self.paths


@dataclass(slots=True)
class GallerySource:
    name: str
    prefix: str
    dataset: FaceDataset
    dataset_root: Path
    images_root: Path
    identity_lookup: DatasetIdentityLookup
    cache_dir: Path


class CompositeIdentityLookup(DatasetIdentityLookup):
    def __init__(
        self,
        mapping: Dict[str, DatasetIdentityLookup],
        default_lookup: Optional[DatasetIdentityLookup] = None,
    ) -> None:
        self.mapping = mapping
        self.default_lookup = default_lookup

    def lookup(self, file_key: str) -> str:
        if "___" in file_key:
            prefix, _ = file_key.split("___", maxsplit=1)
            lookup = self.mapping.get(prefix)
            if lookup is None:
                if self.default_lookup is None:
                    raise KeyError(
                        f"No identity lookup registered for prefix '{prefix}'"
                    )
                return self.default_lookup.lookup(file_key)
            return lookup.lookup(file_key)
        if self.default_lookup is None:
            raise KeyError("Composite identity lookup requires prefixed keys")
        return self.default_lookup.lookup(file_key)


def resolve_dataset(
    dataset_name: Literal["CelebA", "lfw", "NeRSembleGT"],
    celeba_test_set_only: bool = False,
) -> DatasetSpec:
    if dataset_name == "CelebA":
        root = DATASETS_DIR / "CelebA"
        identity_lookup = CelebAIdentityLookup(
            root / "identity_CelebA.txt", test_set_only=celeba_test_set_only
        )
        return DatasetSpec(
            name="CelebA",
            root=root,
            images_subdir=Path("img_align_celeba"),
            file_extension=".jpg",
            identity_lookup=identity_lookup,
            celeba_test_set_only=celeba_test_set_only,
        )
    if dataset_name == "lfw":
        root = DATASETS_DIR / "lfw"
        identity_lookup = LFWIdentityLookup()
        return DatasetSpec(
            name="lfw",
            root=root,
            images_subdir=Path("lfw-deepfunneled"),
            file_extension=".jpg",
            identity_lookup=identity_lookup,
        )
    if dataset_name in ["NeRSembleGT", "NeRSembleReconst"]:
        root = DATASETS_DIR / dataset_name
        if not root.exists():
            raise FileNotFoundError(
                f"Dataset root {root} not found. Ensure {dataset_name} is placed under datasets/."
            )
        identity_lookup = NeRSembleIdentityLookup()
        return DatasetSpec(
            name=dataset_name,
            root=root,
            images_subdir=(
                Path("images") if dataset_name == "NeRSembleGT" else Path("renders")
            ),
            file_extension=".png",
            identity_lookup=identity_lookup,
        )
    raise ValueError(f"Unsupported dataset {dataset_name}")
