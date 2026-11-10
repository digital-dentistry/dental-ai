"""파일 읽기·쓰기의 Input / Output.  모든 구현(<원래 형식>_to_<새 형식>.py)이 이 두 struct를 받는다.

    out = IOOutput()
    if not npz_to_ply(IOInput(Path("cache/IOSSEG007_upper.npz"), Path("IOSSEG007_upper.ply")), out):
        ...                                     # 바꿀 수 없는 파일이다.  out을 읽지 않는다
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class IOInput:
    src: Path                                # 읽을 파일
    dst: Path                                # 쓸 파일.  폴더가 없으면 만들고, 있는 파일은 덮어쓴다


@dataclass
class IOOutput:
    size: int = 0                            # 쓴 바이트 수
