"""데이터 받기의 Input / Output.  모든 구현(from_<저장소>.py)이 이 두 struct를 받는다.

구현이 False를 내면 받지 못한 것이다: out을 읽지 않는다 (이유는 구현이 stderr에 쓴다).

    ddh = DownloadDataOutput()
    if from_ddh(DownloadDataInput("ios-teethseg-full", "v1"), ddh):
        train = ToothSegDataset([ddh.cache], ddh.train, ToothSegConfig())
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DownloadDataInput:
    dataset: str                         # 데이터셋 이름 (예: ios-teethseg-full)
    version: str                         # 릴리스 버전 (예: v1)
    out_dir: Path = Path("data/ios")     # 이 아래 <저장소>_<데이터셋>_<버전>/에 받는다.  상대 경로는 지금 디렉터리 기준
    client: Any = None                   # 저장소 접속 (from_ddh: boto3 S3 client).  None이면 환경 변수로 만든다


@dataclass
class DownloadDataOutput:
    root: Path | None = None             # 받은 폴더.  README.md (데이터 카드)가 여기 있다
    cache: Path | None = None            # ToothSegDataset의 cache_dirs에 넣는다
    train: list[str] = field(default_factory=list)   # 스캔 이름 (캐시 파일 이름).  저장소가 환자 단위로 정한 split
    val: list[str] = field(default_factory=list)
    test: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)         # 릴리스 정보.  내용은 구현마다 다르다 (from_ddh: meta.json)
