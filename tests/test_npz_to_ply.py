"""npz_to_ply: 캐시 -> PLY -> ply_decoder_cpu로 다시 읽으면 캐시와 같다."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from dental_ai.dataset.ToothSegPreProcessing import fdi_to_class, save_cache
from dental_ai.dataset.algorithm.augmentation.augmentation import UNKNOWN_TOOTH
from dental_ai.dataset.algorithm.scan_decoder.ply_decoder_cpu import ply_decoder_cpu
from dental_ai.dataset.algorithm.scan_decoder.scan_decoder import ScanDecoderInput, ScanDecoderOutput, VertexLabel
from dental_ai.dataset.algorithm.io.npz_to_ply import classes_to_fdi, npz_to_ply
from dental_ai.dataset.algorithm.io.io import IOInput, IOOutput


def make_cache(path: Path, jaw_upper: bool, normals: bool = True, colour: bool = True) -> dict[str, np.ndarray]:
    """class 0..17이 모두 있는 캐시."""
    rng = np.random.default_rng(1)
    n = 18 * 20
    xyz = rng.uniform(-30.0, 30.0, (n, 3)).astype(np.float32)
    nrm = rng.normal(size=(n, 3)).astype(np.float32)
    nrm = nrm / np.linalg.norm(nrm, axis=1, keepdims=True) if normals else np.zeros((n, 3), np.float32)
    cls = np.repeat(np.arange(18), 20).astype(np.uint8)
    ins = np.where(cls > 0, cls, 0).astype(np.uint8)
    rgb = rng.integers(0, 256, (n, 3)).astype(np.uint8) if colour else None
    save_cache(path, torch.from_numpy(xyz), torch.from_numpy(nrm), cls, ins, jaw_upper, rgb)
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


def decode(path: Path) -> ScanDecoderOutput:
    mesh = ScanDecoderOutput()
    assert ply_decoder_cpu(ScanDecoderInput(path), mesh)
    return mesh


@pytest.mark.parametrize("jaw_upper", [True, False])
def test_ply_reads_back_as_the_cache(tmp_path: Path, jaw_upper: bool) -> None:
    cache = make_cache(tmp_path / "a.npz", jaw_upper)
    out = IOOutput()
    assert npz_to_ply(IOInput(tmp_path / "a.npz", tmp_path / "ply" / "a.ply"), out)
    assert out.size == (tmp_path / "ply" / "a.ply").stat().st_size

    mesh = decode(tmp_path / "ply" / "a.ply")
    np.testing.assert_array_equal(mesh.positions.numpy(), cache["xyz"])
    np.testing.assert_array_equal(mesh.normals.numpy(), cache["nrm"].astype(np.float32))
    np.testing.assert_array_equal(np.round(mesh.colours.numpy() * 255).astype(np.uint8), cache["rgb"])
    np.testing.assert_array_equal(mesh.instances.numpy(), cache["ins"])
    # 전처리 규칙으로 다시 class를 만들면 원래 cls다
    labels, fdi = mesh.labels.numpy(), mesh.fdi.numpy()
    cls = fdi_to_class(np.where(labels == VertexLabel.TOOTH, fdi, 0), jaw_upper)
    cls[(labels == VertexLabel.TOOTH) & (cls == 0)] = UNKNOWN_TOOTH
    np.testing.assert_array_equal(cls, cache["cls"])
    assert set(fdi[cache["cls"] == 1].tolist()) == {11 if jaw_upper else 31}
    assert set(fdi[cache["cls"] == 9].tolist()) == {21 if jaw_upper else 41}
    header = (tmp_path / "ply" / "a.ply").read_bytes().split(b"end_header")[0].decode()
    assert f"comment jaw {'upper' if jaw_upper else 'lower'}" in header
    assert "element face 0" in header


def test_no_normals_and_no_colour_are_left_out(tmp_path: Path) -> None:
    make_cache(tmp_path / "a.npz", True, normals=False, colour=False)
    assert npz_to_ply(IOInput(tmp_path / "a.npz", tmp_path / "a.ply"), IOOutput())
    header = (tmp_path / "a.ply").read_bytes().split(b"end_header")[0].decode()
    assert "nx" not in header and "red" not in header
    mesh = decode(tmp_path / "a.ply")
    assert mesh.normals is None and mesh.colours is None


def test_classes_to_fdi() -> None:
    cls = np.array([0, 1, 8, 9, 16, UNKNOWN_TOOTH])
    assert classes_to_fdi(cls, True).tolist() == [0, 11, 18, 21, 28, 0]
    assert classes_to_fdi(cls, False).tolist() == [0, 31, 38, 41, 48, 0]


def test_files_that_are_not_a_cache_are_false(tmp_path: Path) -> None:
    (tmp_path / "text.npz").write_text("not a zip")
    np.savez(tmp_path / "other.npz", points=np.zeros((3, 3)))
    np.savez(tmp_path / "bad_class.npz", xyz=np.zeros((2, 3), np.float32), nrm=np.zeros((2, 3), np.float16),
             cls=np.array([0, 18], np.uint8), ins=np.zeros(2, np.uint8), jaw=np.uint8(1))
    for name in ("text", "other", "bad_class"):
        assert not npz_to_ply(IOInput(tmp_path / f"{name}.npz", tmp_path / f"{name}.ply"), IOOutput())
        assert not (tmp_path / f"{name}.ply").exists()
