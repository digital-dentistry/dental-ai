"""스캔 encoder의 Input / Output."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class ScanEncoderInput:
    path: Path
    positions: torch.Tensor                  # [V,3] float32 mm
    faces: torch.Tensor | None = None        # [F,3]
    colours: torch.Tensor | None = None      # [V,3] float32 0..1 — 실제 정점 색일 때만
    # 정점 label — 셋 다 주거나 셋 다 None (label 없는 스캔)
    fdi: torch.Tensor | None = None          # [V] 치아 번호 11..48, 위치가 없으면 0
    labels: torch.Tensor | None = None       # [V] scan_decoder.VertexLabel (-1 미분류, 0 치은, 1 치아, 2 스캔바디 ...)
    instances: torch.Tensor | None = None    # [V] 치아 instance 번호 1..K, 치아가 아니면 0
    comments: tuple[str, ...] = ()           # header comment 줄 (출처 등)
    # 위치 인자로 만드는 곳이 있어 맨 뒤에 둔다.  data/ios raw에는 없다 (전처리가 face에서 다시 구한다)
    normals: torch.Tensor | None = None      # [V,3] float32.  주면 nx ny nz로 쓴다


@dataclass
class ScanEncoderOutput:
    size: int = 0                            # 쓴 바이트 수
