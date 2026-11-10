"""치아 instance segmentation 학습용 Dataset / DataLoader.

segmentation3d/training/dataset.py(TeethDataset)를 network와 떼어 낸 포트다.  한 item은
full-resolution 캐시 스캔을 읽어 augmentation -> voxel downsample -> 원점 중심 정렬 ->
feature·target 순으로 만든다.  알고리즘은 dataset/algorithm/에 있고, augmentation과
downsample은 RNG를 뽑는 순서까지 원본과 같아서 같은 seed면 segmentation3d와 같은
점·label·feature가 나온다.

segmentation3d는 dataset 안에서 network 전용 geometry(serialization hierarchy, query seed)
까지 만들었다.  그건 network마다 다르므로 dental_ai.network 몫으로 두고, 이 loader는
점(coord, grid)·feature·label만 낸다.

캐시는 data/ios/<데이터셋>/cache/의 npz다 (ToothSegPreProcessing.py.  Teeth3DS는 segmentation3d 캐시와 같다):
    xyz [V,3] float32 mm, nrm [V,3] float16, cls [V] uint8, ins [V] uint8, jaw uint8 (1 upper),
    rgb [V,3] uint8 (실제 색이 있을 때만 — 지금의 오픈 데이터셋에는 없다)
색이 없는 스캔은 synth_colour_cpu가 in-house 실측 컬러맵으로 치아 / 치은별로 칠한다.
DDH 데이터셋 릴리스 (S3)는 dental_ai ddh-fetch가 같은 형식의 캐시로 받는다 (DDHLoader.py).

    from dental_ai.dataset.ToothSegLoader import (ToothSegConfig, ToothSegDataset, patient_split,
                                                  read_split, tooth_seg_loader)
    IOS = Path("data/ios")
    split = IOS / "teeth3ds/splits/Teeth3DS_train_test_split"
    names = read_split(split / "training_upper.txt") + read_split(split / "training_lower.txt")
    names += read_split(IOS / "3d_iosseg/splits/train.txt") + read_split(IOS / "poseidon3d/splits/train.txt")
    train, val = patient_split(names)
    caches = [IOS / d / "cache" for d in ("teeth3ds", "3d_iosseg", "poseidon3d", "cbct_ios_fusion_liu2023")]
    cfg = ToothSegConfig(features=("xyz", "normal", "jaw", "colour", "curvature"))
    loader = tooth_seg_loader(ToothSegDataset(caches, train, cfg), batch_size=12, workers=12)
    for batch in loader:
        batch = batch.to("cuda", non_blocking=True)        # batch.features [N, cfg.in_ch]

디버그: augmentation이 끝난 item을 opencv로 본다 (맨 아래 render_debug / _debug_main).
    python -m dental_ai.dataset.ToothSegLoader --cache data/ios/teeth3ds/cache
"""
from __future__ import annotations

import zlib
from collections.abc import Callable
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .algorithm.augmentation.augmentation import (CROP_MODES, UNKNOWN_TOOTH, AugmentationInput, AugmentationOutput,
                                                  AugmentationParams, ScanSample)
from .algorithm.augmentation.segment_crop_cpu import segment_crop_cpu
from .algorithm.augmentation.synth_colour_cpu import synth_colour_cpu
from .algorithm.augmentation.train_recipe_cpu import train_recipe_cpu
from .algorithm.curvature.curvature import CurvatureInput, CurvatureOutput
from .algorithm.curvature.local_pca_cpu import local_pca_cpu
from .algorithm.voxel_downsample.voxel_downsample import VoxelDownsampleInput, VoxelDownsampleOutput
from .algorithm.voxel_downsample.voxel_grid_cpu import voxel_grid_cpu

# ------------------------------------------------------------------ labels
NUM_CLASSES = 18                 # 0 gingiva, 1..8 첫 사분면, 9..16 둘째 사분면, 17 위치 모르는 치아(in-house)
NUM_ARCH_POSITIONS = 9
# 좌우를 무시한 악궁 위치: 0 gingiva, 1 중절치 .. 8 제3대구치.  SO(3) augmentation 아래서는
# 좌우를 구별할 수 없어 17-class target은 noise가 된다.  17 -> -1 (cross_entropy ignore_index).
ARCH_POSITION = torch.tensor([0] + list(range(1, 9)) * 2 + [-1])

# ---------------------------------------------------------------- features
ARCH_RADIUS_MM = 35.0            # xyz / 35 mm -> 악궁이 대략 [-1, 1]
# feature 이름 -> channel 수.  config.features 순서대로 이어 붙인다.
# ("xyz", "normal", "jaw")가 segmentation3d의 in_ch 7, + "colour"가 in_ch 10이다.
# "curvature"는 config.curvature_k의 scale 하나당 2 channel.
FEATURE_CHANNELS = {"xyz": 3, "normal": 3, "jaw": 1, "colour": 3, "curvature": 2}


@dataclass(frozen=True)
class ToothSegConfig:
    voxel_mm: float = 0.25
    features: tuple[str, ...] = ("xyz", "normal", "jaw", "colour")
    curvature_k: tuple[int, ...] = (32,)     # "curvature"의 이웃 수 (scale).  0.25 mm voxel에서 k=32 ≈ 반경 0.8 mm
    max_points: int | None = 70000           # downsample 후 이보다 많으면 무작위로 솎는다
    train: bool = True                       # augmentation (train recipe)
    augmentation: AugmentationParams = field(default_factory=AugmentationParams)
    colour_force: bool = False               # 검증: 실제 색, 없으면 칠한 색 (기본은 색 없이)
    crop_force: str | None = None            # 검증: 항상 이 방식으로 crop ("bridge" / ...)

    def __post_init__(self) -> None:
        unknown = [f for f in self.features if f not in FEATURE_CHANNELS]
        if unknown:
            raise ValueError(f"unknown features {unknown}; choose from {list(FEATURE_CHANNELS)}")
        if self.crop_force is not None and self.crop_force not in CROP_MODES:
            raise ValueError(f"crop_force must be one of {CROP_MODES}")
        if self.colour_force and "colour" not in self.features:
            raise ValueError('colour_force needs "colour" in features')

    @property
    def in_ch(self) -> int:
        return sum(FEATURE_CHANNELS[f] * (len(self.curvature_k) if f == "curvature" else 1)
                   for f in self.features)


@dataclass
class ToothSegItem:
    name: str
    coord: torch.Tensor                      # [n,3] float32 mm, 원점 중심
    grid: torch.Tensor                       # [n,3] int64 downsample voxel 좌표
    features: torch.Tensor                   # [n, in_ch]
    classes: torch.Tensor                    # [n] 0..17
    instances: torch.Tensor                  # [n] 0 gingiva, 1..G
    arch_positions: torch.Tensor             # [n] 0..8, -1 unknown
    offset_targets: torch.Tensor             # [n,3] 치아 점 -> 그 치아 centroid (gingiva 0)
    jaw_upper: bool


@dataclass
class ToothSegBatch:
    """스캔들을 점 단위로 이어 붙인 batch.  instances id는 스캔마다 따로다 (loss가 스캔별로 묶는다)."""

    names: list[str]
    counts: list[int]                        # 스캔마다 점 수
    coord: torch.Tensor                      # [N,3]
    grid: torch.Tensor                       # [N,3]
    features: torch.Tensor                   # [N, in_ch]
    classes: torch.Tensor                    # [N]
    instances: torch.Tensor                  # [N]
    arch_positions: torch.Tensor             # [N]
    offset_targets: torch.Tensor             # [N,3]
    jaw_upper: torch.Tensor                  # [B] bool

    @property
    def batch_index(self) -> torch.Tensor:
        """[N] 각 점이 속한 스캔 번호."""
        dev = self.coord.device
        return torch.repeat_interleave(torch.arange(len(self.counts), device=dev),
                                       torch.tensor(self.counts, device=dev))

    def to(self, device: torch.device | str, non_blocking: bool = False) -> ToothSegBatch:
        return self._map(lambda t: t.to(device, non_blocking=non_blocking))

    def pin_memory(self) -> ToothSegBatch:
        """DataLoader(pin_memory=True)가 부른다."""
        return self._map(torch.Tensor.pin_memory)

    def _map(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> ToothSegBatch:
        return replace(self, **{f.name: fn(getattr(self, f.name)) for f in fields(self)
                                if isinstance(getattr(self, f.name), torch.Tensor)})


# ---------------------------------------------------------------- features
def unit_normals(normals: torch.Tensor) -> torch.Tensor:
    return normals / torch.linalg.vector_norm(normals, dim=1, keepdim=True).clamp_min(1e-9)


def standardise_colour(rgb: torch.Tensor) -> torch.Tensor:
    """스캔별로 표준화한 RGB.  스캐너마다 white balance와 노출이 달라 상대적인 색만 의미가
    있다.  STL은 색이 없고 Teeth3DS OBJ는 ~0.004 양자화 흔들림뿐인 회색이라 둘 다 정확히
    0 — network의 "정보 없음" 값 — 이 된다."""
    sd = rgb.std(0, correction=0)
    if float(sd.mean()) < 0.01:
        return torch.zeros_like(rgb)
    return ((rgb - rgb.mean(0)) / sd.clamp_min(1e-6)).clamp(-4.0, 4.0)


def point_features(features: tuple[str, ...], coord: torch.Tensor, normals: torch.Tensor, jaw_upper: bool,
                   colour: torch.Tensor | None = None, curvature: torch.Tensor | None = None) -> torch.Tensor:
    """`features` 순서대로 이어 붙인 [n, in_ch].  추론도 이 함수를 써야 학습과 입력이 같다.

    coord: 원점 중심 mm.  normals: unit length.  colour: 0..1 RGB 또는 None (0으로 채움).
    curvature: local_pca_cpu()의 결과 ("curvature"가 있을 때만).
    """
    cols = []
    for f in features:
        if f == "xyz":
            cols.append(coord / ARCH_RADIUS_MM)
        elif f == "normal":
            cols.append(normals)
        elif f == "jaw":
            cols.append(torch.full((coord.shape[0], 1), 1.0 if jaw_upper else 0.0))
        elif f == "colour":
            cols.append(standardise_colour(colour) if colour is not None else torch.zeros_like(coord))
        elif f == "curvature":
            cols.append(curvature)
    return torch.cat(cols, 1)


# ----------------------------------------------------------------- targets
def arch_position(classes: torch.Tensor) -> torch.Tensor:
    """[n] class 0..17 -> [n] 악궁 위치 0..8, -1 unknown."""
    return ARCH_POSITION[classes]


def offset_targets(coord: torch.Tensor, instances: torch.Tensor) -> torch.Tensor:
    """[n,3] 치아 점에서 그 치아 centroid까지의 offset (gingiva 0)."""
    out = torch.zeros_like(coord)
    tooth = torch.nonzero(instances > 0).squeeze(1)
    ids, slot = torch.unique(instances[tooth], return_inverse=True)
    centroid = torch.zeros(ids.numel(), 3, dtype=torch.float64)
    centroid.index_add_(0, slot, coord[tooth].double())
    centroid /= torch.bincount(slot, minlength=ids.numel())[:, None]
    out[tooth] = (centroid[slot] - coord[tooth].double()).to(out.dtype)
    return out


# ----------------------------------------------------------------- dataset
def _name_generator(name: str) -> torch.Generator:
    """검증의 draw(강제 crop, 칠한 색, 솎기)는 스캔마다 고정한다."""
    return torch.Generator().manual_seed(zlib.crc32(name.encode()))


class ToothSegDataset(Dataset):
    """npz 캐시 스캔 -> ToothSegItem.  모든 처리는 host(DataLoader worker)에서 돈다."""

    def __init__(self, cache_dirs: list[Path | str], names: list[str], config: ToothSegConfig | None = None) -> None:
        self.config = config or ToothSegConfig()
        # 색을 안 쓰면 칠하지도 않는다 (colour_p=0은 RNG를 뽑지 않는다)
        aug = self.config.augmentation
        self._params = aug if "colour" in self.config.features else replace(aug, colour_p=0.0)
        # normal 없는 DDH 릴리스의 캐시는 nrm이 전부 0이다.  이 feature들이 조용히 0이 되면 안 된다
        self._needs_normals = "normal" in self.config.features or "curvature" in self.config.features
        # 캐시 여러 개를 섞을 수 있다 (Teeth3DS + in-house).  이름은 그걸 가진 첫 캐시에서 읽는다.
        self._path: dict[str, Path] = {}
        for n in names:
            for d in map(Path, cache_dirs):
                if (d / f"{n}.npz").exists():
                    self._path[n] = d / f"{n}.npz"
                    break
        self.names = [n for n in names if n in self._path]
        self.missing = [n for n in names if n not in self._path]

    def __len__(self) -> int:
        return len(self.names)

    def load(self, name: str) -> tuple[ScanSample, bool]:
        with np.load(self._path[name]) as d:
            nrm = d["nrm"]
            if self._needs_normals and not nrm.any():
                raise ValueError(f'{name}: normal이 없는 캐시다.  features에서 "normal", "curvature"를 뺀다')
            rgb = torch.from_numpy(d["rgb"].astype(np.float32) / 255.0) if "rgb" in d.files else None
            sample = ScanSample(torch.from_numpy(d["xyz"].astype(np.float32)),
                                torch.from_numpy(nrm.astype(np.float32)),
                                torch.from_numpy(d["cls"].astype(np.int64)),
                                torch.from_numpy(d["ins"].astype(np.int64)), rgb)
            return sample, bool(d["jaw"])

    def __getitem__(self, i: int) -> ToothSegItem:
        cfg = self.config
        name = self.names[i]
        sample, jaw_upper = self.load(name)
        # 학습은 worker의 torch RNG (DataLoader가 worker마다 다르게 seed), 검증은 스캔마다 고정
        gen = None if cfg.train else _name_generator(name)

        aug = AugmentationOutput()
        if cfg.crop_force and segment_crop_cpu(
                AugmentationInput(sample, replace(self._params, crop_mode=cfg.crop_force), gen), aug):
            sample = aug.sample
        if cfg.train:
            if train_recipe_cpu(AugmentationInput(sample, self._params, gen), aug):
                sample = aug.sample
        elif not cfg.colour_force:
            sample = sample.replace(colour=None)                 # 검증 기본: 색 없이
        elif sample.colour is None and synth_colour_cpu(AugmentationInput(sample, self._params, gen), aug):
            sample = aug.sample

        ds = VoxelDownsampleOutput()
        if not voxel_grid_cpu(VoxelDownsampleInput(sample.positions, cfg.voxel_mm), ds):
            raise ValueError(f"{name}: nothing to downsample")
        rep = ds.representative
        # 곡률은 솎기 전에, jitter 없는 표면에서 구한다: 이웃 밀도가 voxel 간격 그대로여야
        # k의 물리적 크기가 일정하고, jitter(0.05 mm)는 bend를 noise로 덮는다 (치은 경계
        # AUC 0.895 -> 0.817).  솎지도 jitter하지도 않는 추론과 같은 값이 된다.
        curvature = None
        if "curvature" in cfg.features:
            cv = CurvatureOutput()
            if not local_pca_cpu(CurvatureInput(sample.surface[rep], unit_normals(sample.normals[rep]),
                                                cfg.curvature_k), cv):
                raise ValueError(f"{name}: could not compute curvature")
            curvature = cv.curvature
        keep = torch.arange(rep.numel())
        if cfg.max_points and keep.numel() > cfg.max_points:
            keep = torch.randperm(keep.numel(), generator=gen)[:cfg.max_points]
        rep = rep[keep]
        # 추론처럼 원점 중심: 배울 이동이 없다
        coord = sample.positions[rep] - sample.positions[rep].double().mean(0).float()
        normals = unit_normals(sample.normals[rep])
        classes, instances = sample.classes[rep], sample.instances[rep]
        features = point_features(cfg.features, coord, normals, jaw_upper,
                                  None if sample.colour is None else sample.colour[rep],
                                  None if curvature is None else curvature[keep])
        return ToothSegItem(name, coord, ds.cells[keep], features, classes, instances, arch_position(classes),
                            offset_targets(coord, instances), jaw_upper)


def collate(items: list[ToothSegItem]) -> ToothSegBatch:
    """스캔들을 한 batch로: 점은 이어 붙이고 스캔마다 점 수(counts)를 남긴다."""
    return ToothSegBatch(
        names=[it.name for it in items], counts=[it.coord.shape[0] for it in items],
        coord=torch.cat([it.coord for it in items]), grid=torch.cat([it.grid for it in items]),
        features=torch.cat([it.features for it in items]), classes=torch.cat([it.classes for it in items]),
        instances=torch.cat([it.instances for it in items]),
        arch_positions=torch.cat([it.arch_positions for it in items]),
        offset_targets=torch.cat([it.offset_targets for it in items]),
        jaw_upper=torch.tensor([it.jaw_upper for it in items]))


# ------------------------------------------------------------ split / loader
def read_split(path: Path | str) -> list[str]:
    return [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()]


def patient_split(names: list[str], val_frac: float = 0.1, seed: int = 0) -> tuple[list[str], list[str]]:
    """환자 단위 (train, val) 분할.  이름은 '<patient>_<jaw>'.

    한 환자의 upper와 lower가 양쪽에 갈라지면 val 점수가 높게 나온다.  스캔이 몇 개
    안 돼서 val이 비면 train 전체로 채점한다.
    """
    rng = np.random.default_rng(seed)
    by_pid: dict[str, list[str]] = {}
    for n in names:
        by_pid.setdefault(n.split("_")[0], []).append(n)
    pids = sorted(by_pid)
    rng.shuffle(pids)
    want, val = int(len(names) * val_frac), []
    for p in pids:
        if len(val) >= want:
            break
        val += by_pid[p]
    if not val:
        return list(names), list(names)
    held = set(val)
    return [n for n in names if n not in held], val


def _worker_init(_: int) -> None:
    torch.set_num_threads(1)                     # worker가 여럿이니 worker마다 1 thread


def tooth_seg_loader(dataset: ToothSegDataset, batch_size: int, workers: int = 8, shuffle: bool | None = None,
                     drop_last: bool | None = None, pin_memory: bool = True) -> DataLoader:
    """학습 config면 shuffle + drop_last, 검증 config면 둘 다 끈다 (segmentation3d train.py와 같다)."""
    train = dataset.config.train
    return DataLoader(dataset, batch_size=batch_size, shuffle=train if shuffle is None else shuffle,
                      drop_last=train if drop_last is None else drop_last, num_workers=workers,
                      collate_fn=collate, worker_init_fn=_worker_init, pin_memory=pin_memory,
                      persistent_workers=workers > 0, prefetch_factor=3 if workers else None)


# -------------------------------------------------------------- debug view
# augmentation이 끝난 item (network가 받는 그대로)을 opencv로 본다.  디버그 전용이라 opencv는
# 의존성이 아니다: pip install opencv-python (화면 없는 서버는 opencv-python-headless + --save).
#
#   python -m dental_ai.dataset.ToothSegLoader --cache data/ios/teeth3ds/cache
#   python -m dental_ai.dataset.ToothSegLoader --cache data/ios/3d_iosseg/cache --save runs/aug_debug --count 20
#
# 창 키: n / space 다음, p 이전, r 같은 스캔을 다시 augment, v 시점 전환, s png 저장, q / esc 종료.
DEBUG_BACKGROUND = (0.102, 0.102, 0.098)         # #1a1a19
DEBUG_TEXT = (0.941, 0.937, 0.925)               # #f0efec
DEBUG_GINGIVA = (0.62, 0.61, 0.58)
DEBUG_CONVEX, DEBUG_CONCAVE = (0.165, 0.471, 0.839), (0.922, 0.408, 0.204)    # #2a78d6, #eb6834


def feature_slices(config: ToothSegConfig) -> dict[str, slice]:
    """feature 이름 -> item.features에서 그 feature의 열 범위 (예: "normal" -> slice(3, 6))."""
    out, c = {}, 0
    for f in config.features:
        w = FEATURE_CHANNELS[f] * (len(config.curvature_k) if f == "curvature" else 1)
        out[f], c = slice(c, c + w), c + w
    return out


def _view_frame(coord: np.ndarray, normals: np.ndarray | None, classes: np.ndarray, arch: np.ndarray,
                view: str) -> np.ndarray:
    """[3,3] 행이 화면 x(오른쪽), y(위), z(카메라 쪽)인 회전.

    "occlusal": 교합면을 정면으로 (z = 치아 normal 평균), 악궁 위치 번호가 작아지는 쪽
    (전치 쪽)이 위.  x = y × z인 진짜 회전이라 화면이 뒤집히지 않는다 — 전체 악궁이면
    label이 맞을 때 위·아래턱 모두 class 1..8이 왼쪽, 9..16이 오른쪽에 온다 (mirror
    augmentation 확인용).  치아 몇 개만 남은 crop은 근심 쪽이 위로 온다.
    "raw": 좌표 그대로 xy (회전 augmentation이 보인다).
    """
    if view == "raw":
        return np.eye(3)
    pca = np.linalg.svd(coord - coord.mean(0), full_matrices=False)[2]
    tooth = classes > 0
    z = normals[tooth].mean(0) if normals is not None and tooth.any() else pca[2]
    z = z / max(np.linalg.norm(z), 1e-9)
    # 위치 번호(1 중절치 .. 8 제3대구치)가 커지는 방향의 반대 = 전치 / 근심 쪽
    t = arch >= 1
    if t.any() and np.ptp(arch[t]) > 0:
        y = -((arch[t] - arch[t].mean())[:, None] * (coord[t] - coord[t].mean(0))).sum(0)
    else:
        y = pca[0]
    y = y - (y @ z) * z
    if np.linalg.norm(y) < 1e-6:
        y = pca[1] - (pca[1] @ z) * z
    y = y / np.linalg.norm(y)
    return np.stack([np.cross(y, z), y, z])


def _splat(u: np.ndarray, v: np.ndarray, depth: np.ndarray, rgb: np.ndarray, size: int, radius: int) -> np.ndarray:
    """점을 (2r+1)² 칸으로 찍고 픽셀마다 카메라에 가장 가까운 점만 남긴다 (z-buffer)."""
    img = np.empty((size, size, 3), np.float32)
    img[:] = DEBUG_BACKGROUND
    r = range(-radius, radius + 1)
    uu = np.concatenate([u + dx for dy in r for dx in r])
    vv = np.concatenate([v + dy for dy in r for dx in r])
    dd, cc = np.tile(depth, len(r) ** 2), np.tile(rgb, (len(r) ** 2, 1))
    ok = (uu >= 0) & (uu < size) & (vv >= 0) & (vv < size)
    pix, dd, cc = vv[ok] * size + uu[ok], dd[ok], cc[ok]
    order = np.lexsort((dd, pix))                                # 픽셀별로 모으고 그 안에서 depth 오름차순
    last = np.r_[pix[order][1:] != pix[order][:-1], True]       # 픽셀마다 가장 가까운 점
    img.reshape(-1, 3)[pix[order][last]] = cc[order][last]
    return img


def _palette(i: int) -> tuple[float, float, float]:
    """instance id -> 고정된 색 (golden angle hue)."""
    import colorsys
    return colorsys.hsv_to_rgb((i * 0.618034) % 1.0, 0.65, 0.95)


def _class_colour(c: int) -> tuple[float, float, float]:
    """악궁 위치는 hue, 사분면은 밝기 (1..8 밝게, 9..16 진하게) — mirror가 틀리면 바로 보인다."""
    import colorsys
    if c == 0:
        return DEBUG_GINGIVA
    if c == UNKNOWN_TOOTH:
        return (0.05, 0.05, 0.05)
    pos, second = (c - 1) % 8, c > 8
    return colorsys.hsv_to_rgb(pos / 8.0, 0.95 if second else 0.5, 0.7 if second else 1.0)


def render_debug(item: ToothSegItem, config: ToothSegConfig, view: str = "occlusal", panel: int = 400,
                 title: str = "") -> np.ndarray:
    """item 하나 -> BGR uint8 그림 (3 x 2 패널: 색, instance, class, offset, normal, 곡률)."""
    import cv2

    sl = feature_slices(config)
    feats = item.features.numpy()
    coord, classes, ins = item.coord.double().numpy(), item.classes.numpy(), item.instances.numpy()
    normals = feats[:, sl["normal"]].astype(np.float64) if "normal" in sl else None
    frame = _view_frame(coord, normals, classes, item.arch_positions.numpy(), view)
    q = coord @ frame.T
    lo, hi = q[:, :2].min(0), q[:, :2].max(0)
    scale = 0.9 * panel / max(float((hi - lo).max()), 1e-6)
    centre = (lo + hi) / 2
    u = np.round((q[:, 0] - centre[0]) * scale + panel / 2).astype(np.int64)
    v = np.round(panel / 2 - (q[:, 1] - centre[1]) * scale).astype(np.int64)
    # 점 간격이 1~2 픽셀을 넘으면 크게 찍는다 (3D-IOSSeg는 Teeth3DS보다 10배 성기다)
    spacing = np.sqrt(float(np.prod(hi - lo)) * 0.4 / len(q)) * scale
    radius = int(np.clip(round(0.6 * spacing), 1, 4))
    nv = normals @ frame.T if normals is not None else None
    shade = (0.45 + 0.55 * np.abs(nv[:, 2]))[:, None] if nv is not None else np.ones((len(q), 1))

    def to_px(p: np.ndarray) -> tuple[int, int]:
        p = p @ frame.T
        return int(round((p[0] - centre[0]) * scale + panel / 2)), int(round(panel / 2 - (p[1] - centre[1]) * scale))

    def text(img: np.ndarray, s: str, org: tuple[int, int], size: float = 0.45, centred: bool = False) -> None:
        """어두운 상자 위 글자 (외곽선은 opencv 버전마다 글자 폭이 달라 잔상이 남는다)."""
        (w, h), base = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, size, 1)
        x, y = (org[0] - w // 2, org[1] + h // 2) if centred else org
        cv2.rectangle(img, (x - 2, y - h - 2), (x + w + 1, y + base), (25, 26, 26), -1)
        cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, tuple(int(255 * c) for c in DEBUG_TEXT[::-1]), 1,
                    cv2.LINE_AA)

    def finish(rgb_img: np.ndarray, name: str) -> np.ndarray:
        img = np.ascontiguousarray((np.clip(rgb_img, 0, 1)[:, :, ::-1] * 255).astype(np.uint8))
        text(img, name, (8, 18))
        return img

    def blank(name: str, note: str) -> np.ndarray:
        img = np.empty((panel, panel, 3), np.float32)
        img[:] = DEBUG_BACKGROUND
        img = finish(img, name)
        text(img, note, (8, panel // 2))
        return img

    teeth = [i for i in np.unique(ins) if i > 0]
    centroid = {i: (coord + item.offset_targets.double().numpy())[ins == i].mean(0) for i in teeth}
    tooth_cls = {i: int(np.bincount(classes[ins == i]).argmax()) for i in teeth}

    # 1) 색: network가 받는 표준화 색, ±3 SD -> 0..1
    if "colour" in sl:
        col = feats[:, sl["colour"]]
        colourless = not np.any(col)
        p1 = finish(_splat(u, v, q[:, 2], np.clip(0.5 + col / 6.0, 0, 1) * (shade if colourless else 1), panel,
                           radius), "colour (standardised input)")
        if colourless:
            text(p1, "colourless: all channels 0", (8, 38))
    else:
        p1 = blank("colour", '"colour" not in features')

    # 2) instance: id마다 색, 치아 중심에 id
    rgb = np.array([DEBUG_GINGIVA if i == 0 else _palette(int(i)) for i in range(int(ins.max()) + 1)])[ins]
    p2 = finish(_splat(u, v, q[:, 2], rgb * shade, panel, radius), f"instances ({len(teeth)} teeth)")
    for i in teeth:
        text(p2, str(i), to_px(centroid[i]), 0.4, centred=True)

    # 3) class: 위치 hue / 사분면 밝기, 치아 중심에 class 번호
    rgb = np.array([_class_colour(c) for c in range(NUM_CLASSES)])[classes]
    p3 = finish(_splat(u, v, q[:, 2], rgb * shade, panel, radius),
                "classes (full arch ok: 1-8 left, 9-16 right)" if view == "occlusal" else "classes")
    for i in teeth:
        text(p3, str(tooth_cls[i]), to_px(centroid[i]), 0.4, centred=True)

    # 4) offset target: 치아 점 일부에서 그 치아 중심으로
    rgb = np.where((ins > 0)[:, None], 0.75, 0.45) * np.ones(3)
    p4 = finish(_splat(u, v, q[:, 2], rgb * shade, panel, radius), "offset targets -> tooth centroid")
    pick = np.random.default_rng(0).permutation(np.nonzero(ins > 0)[0])[:400]
    for j in pick:
        colour = tuple(int(255 * c) for c in _palette(int(ins[j]))[::-1])
        cv2.line(p4, (int(u[j]), int(v[j])), to_px(centroid[ins[j]]), colour, 1, cv2.LINE_AA)
    for i in teeth:
        cv2.circle(p4, to_px(centroid[i]), 3, (255, 255, 255), -1, cv2.LINE_AA)

    # 5) normal: 화면 좌표계 기준 0.5 + 0.5 n (파랑 = 카메라 쪽)
    p5 = (finish(_splat(u, v, q[:, 2], 0.5 + 0.5 * nv, panel, radius), "normals (view frame)") if nv is not None
          else blank("normals", '"normal" not in features'))

    # 6) 곡률: 첫 scale의 bend, 파랑 볼록 / 회색 0 / 주황 오목
    if "curvature" in sl:
        bend = np.clip(feats[:, sl["curvature"]][:, 1] / 0.4, -1, 1)[:, None]
        rgb = np.where(bend < 0, np.array(DEBUG_GINGIVA) + (np.array(DEBUG_CONVEX) - DEBUG_GINGIVA) * -bend,
                       np.array(DEBUG_GINGIVA) + (np.array(DEBUG_CONCAVE) - DEBUG_GINGIVA) * bend)
        p6 = finish(_splat(u, v, q[:, 2], rgb, panel, radius),
                    f"curvature bend k={config.curvature_k[0]} (blue convex, orange concave)")
    else:
        p6 = blank("curvature", '"curvature" not in features')

    grid = np.vstack([np.hstack([p1, p2, p3]), np.hstack([p4, p5, p6])])
    header = np.empty((44, grid.shape[1], 3), np.uint8)
    header[:] = tuple(int(255 * c) for c in DEBUG_BACKGROUND[::-1])
    text(header, f"{title}  {item.name}  {'upper' if item.jaw_upper else 'lower'}  {len(q)} pts  "
                 f"in_ch {config.in_ch}  {'train (augmented)' if config.train else 'val'}  view {view}", (8, 18), 0.5)
    text(header, "n/space next   p prev   r re-augment   v view   s save   q quit", (8, 37), 0.42)
    return np.vstack([header, grid])


def _debug_main() -> None:
    import argparse
    import os
    import sys

    ap = argparse.ArgumentParser(description="augmentation이 끝난 item을 opencv로 본다 (디버그용)")
    ap.add_argument("--cache", type=Path, nargs="+", required=True, help="npz 캐시 폴더 (여러 개 가능)")
    ap.add_argument("--names", type=Path, default=None, help="스캔 이름 목록 파일 (기본: 캐시의 모든 npz)")
    ap.add_argument("--features", nargs="+", default=list(FEATURE_CHANNELS), choices=list(FEATURE_CHANNELS))
    ap.add_argument("--val", action="store_true", help="검증 config (augmentation 없이)")
    ap.add_argument("--colour-force", action="store_true", help="검증: 실제 색, 없으면 칠한 색")
    ap.add_argument("--crop-force", choices=CROP_MODES, default=None)
    ap.add_argument("--view", choices=["occlusal", "raw"], default="occlusal")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", type=int, default=0, help="처음 볼 스캔 번호")
    ap.add_argument("--panel", type=int, default=400, help="패널 한 칸 크기 (px)")
    ap.add_argument("--save", type=Path, default=None, help="창 대신 png로 저장 (화면 없는 서버)")
    ap.add_argument("--count", type=int, default=16, help="--save: 저장할 스캔 수")
    a = ap.parse_args()
    import cv2

    names = (read_split(a.names) if a.names else
             sorted({p.stem for d in a.cache for p in Path(d).glob("*.npz")}))
    cfg = ToothSegConfig(features=tuple(a.features), train=not a.val, colour_force=a.colour_force,
                         crop_force=a.crop_force)
    ds = ToothSegDataset(a.cache, names, cfg)
    if not len(ds):
        raise SystemExit(f"no scans found in {[str(d) for d in a.cache]}")
    torch.manual_seed(a.seed)
    if a.save is None and sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        a.save = Path("aug_debug")
        print(f"no display: saving png to {a.save}/ instead (pass --save to choose)")

    if a.save is not None:
        a.save.mkdir(parents=True, exist_ok=True)
        for k in range(min(a.count, len(ds))):
            i = (a.start + k) % len(ds)
            path = a.save / f"{k:03d}_{ds.names[i]}.png"
            cv2.imwrite(str(path), render_debug(ds[i], cfg, a.view, a.panel, f"[{i}/{len(ds)}]"))
            print(path)
        return

    window, i, view, saved = "ToothSegLoader debug", a.start % len(ds), a.view, 0
    item = ds[i]
    while True:
        img = render_debug(item, cfg, view, a.panel, f"[{i}/{len(ds)}]")
        cv2.imshow(window, img)
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key in (ord("n"), ord(" ")):
            i = (i + 1) % len(ds)
            item = ds[i]
        elif key == ord("p"):
            i = (i - 1) % len(ds)
            item = ds[i]
        elif key == ord("r"):
            item = ds[i]                                     # 학습 config면 새 augmentation
        elif key == ord("v"):
            view = "raw" if view == "occlusal" else "occlusal"
        elif key == ord("s"):
            out = Path("aug_debug")
            out.mkdir(exist_ok=True)
            path = out / f"{item.name}_{saved:03d}.png"
            cv2.imwrite(str(path), img)
            saved += 1
            print(path)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    _debug_main()
