from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import torch

from ..augmentation.augmentation import UNKNOWN_TOOTH
from ..scan_decoder.scan_decoder import VertexLabel
from ..scan_encoder.ply_encoder_cpu import ply_encoder_cpu
from ..scan_encoder.scan_encoder import ScanEncoderInput, ScanEncoderOutput
from .io import IOInput, IOOutput

CACHE_KEYS = ("xyz", "nrm", "cls", "ins", "jaw")     # ToothSegPreProcessing.save_cache.  rgb는 있을 때만


def classes_to_fdi(classes: np.ndarray, jaw_upper: bool) -> np.ndarray:
    """cache class -> FDI 번호 (ToothSegPreProcessing.fdi_to_class의 역).  0 치은과 UNKNOWN_TOOTH는 0이다."""
    first, second = (1, 2) if jaw_upper else (3, 4)
    c = classes.astype(np.int64)
    fdi = np.where(c <= 8, first * 10 + c, second * 10 + c - 8)
    fdi[(c == 0) | (c == UNKNOWN_TOOTH)] = 0
    return fdi


def npz_to_ply(inp: IOInput, out: IOOutput) -> bool:
    """npz 캐시 한 파일 (ToothSegPreProcessing, from_ddh가 만든다) -> binary PLY.

    캐시를 읽어 data/ios raw의 정점 label로 되돌리는 것까지가 여기서 하는 일이고, 파일은
    scan_encoder.ply_encoder_cpu가 쓴다.  MeshLab, CloudCompare로 캐시를 보거나 scan_decoder.ply_decoder_cpu로
    다시 읽는 용도다.  캐시에는 face가 없으므로 point cloud가 된다.  정점 속성:

        x y z                 xyz (mm)
        nx ny nz              nrm.  전부 0이면 (normal 없는 DDH 릴리스) 쓰지 않는다
        red green blue        rgb.  있을 때만
        fdi, label            cls와 jaw에서.  label은 0 치은, 1 치아.  cls 17 (위치 모르는 치아)은 fdi 0인 치아다
        instance              ins
        comment jaw <upper|lower>, comment source <npz 이름>

    fdi_to_class(np.where(label == 1, fdi, 0), jaw)에 cls 17 자리를 채우면 원래 cls가 나온다 (전처리 규칙).
    False: npz가 아니거나, 캐시 키가 없거나, 정점이 없거나, 길이가 안 맞거나, cls가 0..17 밖이다.
    """
    src, dst = Path(inp.src), Path(inp.dst)
    if not zipfile.is_zipfile(src):
        return False
    with np.load(src) as d:
        if any(k not in d.files for k in CACHE_KEYS):
            return False
        cache = {k: d[k] for k in d.files}
    xyz, nrm, rgb = cache["xyz"], cache["nrm"], cache.get("rgb")
    classes, instances = cache["cls"].astype(np.int64), cache["ins"].astype(np.int64)
    n = len(xyz)
    if n == 0 or xyz.shape != (n, 3) or nrm.shape != (n, 3) or classes.shape != (n,) or instances.shape != (n,):
        return False
    if (rgb is not None and rgb.shape != (n, 3)) or classes.min() < 0 or classes.max() > UNKNOWN_TOOTH:
        return False

    jaw_upper = bool(cache["jaw"])
    labels = np.where(classes > 0, VertexLabel.TOOTH, VertexLabel.GINGIVA).astype(np.int64)
    enc = ScanEncoderInput(
        dst, torch.from_numpy(xyz.astype(np.float32)),
        colours=None if rgb is None else torch.from_numpy(rgb.astype(np.float32) / 255.0),
        fdi=torch.from_numpy(classes_to_fdi(classes, jaw_upper)),
        labels=torch.from_numpy(labels),
        instances=torch.from_numpy(instances),
        comments=(f"jaw {'upper' if jaw_upper else 'lower'}",
                  f"source {src.name.encode('ascii', 'replace').decode()}"),   # header는 ascii다
        normals=torch.from_numpy(nrm.astype(np.float32)) if nrm.any() else None,
    )
    written = ScanEncoderOutput()
    if not ply_encoder_cpu(enc, written):
        return False
    out.size = written.size
    return True
