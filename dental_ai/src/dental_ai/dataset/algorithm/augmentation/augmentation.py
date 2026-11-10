"""augmentation의 Input / Output.  모든 구현(<name>_cpu.py)이 이 두 struct를 받는다.

augmentation은 downsample 전, full-resolution 스캔에서 돈다 — voxel grid가 epoch마다
다르게 떨어지는 것도 약한 augmentation이 된다.  구현이 False를 내면 (crop 뒤 점이 너무
적다 등) 적용하지 않은 것이다: out을 읽지 말고 원래 sample을 쓴다.

    out = AugmentationOutput()
    if segment_crop_cpu(AugmentationInput(sample, params, gen), out):
        sample = out.sample
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import torch

UNKNOWN_TOOTH = 17               # 치아인데 위치(FDI)를 모름 — in-house 스캔
CROP_MODES = ("bridge", "quadrant", "half")


@dataclass
class AugmentationParams:
    """기본값은 segmentation3d instance 단계(toothseg3d train_inst)가 학습에 쓴 값이다."""

    # rotation: scale과 jitter는 "so3" / "limited" 모두에 적용
    rotation: str = "so3"                    # "so3": SO(3) 전체, "limited": 작은 기울기 + z 회전
    scale: float = 0.1                       # [1 - scale, 1 + scale] uniform
    jitter_mm: float = 0.05                  # 정점별 Gaussian noise
    rot_xy_deg: float = 10.0                 # limited: x, y축 기울기
    rot_z_deg: float = 25.0                  # limited: z축 회전

    # 부분 스캔 crop (wedge / segment)
    crop_mode: str | None = None             # "bridge" / "quadrant" / "half"; None이면 crop_mix에서 뽑는다
    crop_mix: tuple[float, float, float] = (0.4, 0.35, 0.25)
    crop_margin_mm: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "bridge": (1.0, 3.0),                # bridge prep은 좁게 스캔한다
        "quadrant": (1.5, 4.5),
        "half": (2.0, 6.0),                  # 반악궁은 치조제를 더 남긴다
    })
    wedge_keep: tuple[float, float] = (0.30, 0.95)   # wedge가 남기는 한 바퀴의 비율
    crop_min_points: int = 3000              # 이보다 적게 남는 crop은 적용하지 않는다

    # gingiva trim
    trim_reach_mm: tuple[float, float] = (2.5, 14.0)
    trim_min_points: int = 5000

    # 합성 색 (synth_colour_cpu).  colour_edge_mm가 None이면 라벨 경계에서 색이 바로 바뀐다 (예전 동작,
    # segmentation3d와 같은 RNG 순서).  범위를 주면 실제 스캔처럼 경계에서 서서히 바뀐다: 치아 쪽으로 잰
    # 부호 거리의 sigmoid로 치아 색과 치은 색을 섞는다.  스캔마다 범위 안에서 uniform으로 뽑는다.
    colour_edge_mm: tuple[float, float] | None = None       # 반반 섞이는 곳 (치아 안쪽 +, mm)
    colour_edge_width_mm: tuple[float, float] = (1.2, 2.2)  # 10%..90% 폭
    colour_drift_scale: tuple[float, float] = (1.0, 1.0)    # drift와 치아별 차이 SD 배율 (edge 모드에서만)

    # train recipe: 각 단계가 돌 확률
    mirror_p: float = 0.5
    crop_p: float = 0.35
    wedge_p: float = 0.25                    # crop 중 단순 wedge의 비율
    trim_p: float = 0.5
    colour_p: float = 0.6                    # 실제 색을 쓰거나 칠한다; 아니면 색 없이


@dataclass
class ScanSample:
    """full-resolution 라벨 스캔 하나."""

    positions: torch.Tensor                  # [V,3] float32, mm
    normals: torch.Tensor                    # [V,3] float32, unit length, 바깥 방향
    classes: torch.Tensor                    # [V] int64: 0 gingiva, 1..8 첫 사분면, 9..16 둘째, 17 UNKNOWN_TOOTH
    instances: torch.Tensor                  # [V] int64: 0 gingiva, 치아마다 양수 id 하나
    colour: torch.Tensor | None = None       # [V,3] float32 0..1; None = 색 없음
    # [V,3] jitter를 더하기 전 위치 (rotation_cpu가 남긴다).  None이면 positions에 jitter가 없다.
    clean_positions: torch.Tensor | None = None

    @property
    def n(self) -> int:
        return self.positions.shape[0]

    @property
    def surface(self) -> torch.Tensor:
        """jitter 없는 표면 위치.  곡률처럼 jitter에 민감한데 추론에는 jitter가 없는 feature는 이걸로 잰다."""
        return self.positions if self.clean_positions is None else self.clean_positions

    def select(self, keep: torch.Tensor) -> ScanSample:
        return ScanSample(self.positions[keep], self.normals[keep], self.classes[keep], self.instances[keep],
                          None if self.colour is None else self.colour[keep],
                          None if self.clean_positions is None else self.clean_positions[keep])

    def replace(self, **changes: object) -> ScanSample:
        return replace(self, **changes)


@dataclass
class AugmentationInput:
    sample: ScanSample
    params: AugmentationParams = field(default_factory=AugmentationParams)
    # CPU generator.  None이면 torch 전역 RNG — DataLoader가 worker마다 다르게 seed한다.
    generator: torch.Generator | None = None


@dataclass
class AugmentationOutput:
    sample: ScanSample | None = None
