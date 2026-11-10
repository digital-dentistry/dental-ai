"""ddh_viewer: 캐시 -> HTML.  페이지에 넣은 값을 다시 꺼내 캐시와 맞춰 본다 (그리는 것은 브라우저가 한다)."""
from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.spatial import cKDTree

from dental_ai.dataset.ToothSegPreProcessing import save_cache
from dental_ai.dataset.algorithm.download_data.download_data import DownloadDataOutput
from dental_ai.visualizer.ddh_viewer import pca_normals, pick, write_view, write_views


def make_cache(path: Path, jaw_upper: bool = True, normals: bool = True, colour: bool = False) -> dict[str, np.ndarray]:
    """치은 + 치아 3개 (11, 12, 21), 위치 모르는 치아 1개.  점은 악궁처럼 반원에 놓는다."""
    rng = np.random.default_rng(2)
    n = 500
    angle = rng.uniform(0, np.pi, n)
    xyz = np.stack([25 * np.cos(angle), 25 * np.sin(angle), rng.uniform(0, 8, n)], 1).astype(np.float32)
    nrm = np.tile([0.0, 0.0, 1.0], (n, 1)).astype(np.float32) if normals else np.zeros((n, 3), np.float32)
    cls = np.zeros(n, np.uint8)
    ins = np.zeros(n, np.uint8)
    for k, (c, i) in enumerate([(1, 1), (2, 2), (9, 3), (17, 4)]):
        cls[k * 100:(k + 1) * 100], ins[k * 100:(k + 1) * 100] = c, i
    rgb = rng.integers(0, 256, (n, 3)).astype(np.uint8) if colour else None
    save_cache(path, torch.from_numpy(xyz), torch.from_numpy(nrm), cls, ins, jaw_upper, rgb)
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


def page_data(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    return json.loads(re.search(r'<script type="application/json" id="data">(.*?)</script>', text, re.S).group(1))


def decode(b64: str, dtype: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(b64), dtype)


def test_page_holds_the_cache(tmp_path: Path) -> None:
    cache = make_cache(tmp_path / "s.npz", colour=True)
    entry = {"asset_id": "a1", "patient_pid": "p_ab12", "split": "val", "npz_sha256": "x"}
    out = write_view(tmp_path / "s.npz", tmp_path / "html" / "s.html", entry, "ios-teethseg-full/v1")
    d = page_data(out)

    n = len(cache["xyz"])
    pos = decode(d["pos"], "<f4").reshape(n, 3)
    centred = cache["xyz"].astype(np.float64) - cache["xyz"].mean(0)
    np.testing.assert_allclose(np.linalg.norm(pos, axis=1), np.linalg.norm(centred, axis=1), atol=1e-3)   # 돌리기만 했다
    assert np.isclose(d["radius"], np.linalg.norm(pos, axis=1).max())
    sizes = decode(d["size"], "u1")
    assert len(sizes) == n and sizes.min() >= 5                     # 점 지름 0.01 mm 단위
    assert decode(d["cls"], "u1").tolist() == cache["cls"].tolist()
    assert decode(d["ins"], "u1").tolist() == cache["ins"].tolist()
    assert decode(d["rgb"], "u1").reshape(n, 3).tolist() == cache["rgb"].tolist()
    nrm = decode(d["nrm"], "i1").reshape(n, 3) / 127
    assert np.allclose(np.linalg.norm(nrm, axis=1), 1, atol=0.02)
    front = np.array(d["front"]).reshape(3, 3)
    assert np.allclose(front @ front.T, np.eye(3)) and np.isclose(np.linalg.det(front), 1)

    assert [(r["label"], r["count"]) for r in d["legend"]["fdi"]] == [
        ("치은", 100), ("11", 100), ("12", 100), ("21", 100), ("? (위치 모름)", 100)]
    assert [r["label"] for r in d["legend"]["instance"]] == ["치은", "#1  11", "#2  12", "#3  21", "#4  ? (위치 모름)"]
    info = dict(d["info"])
    assert info["릴리스"] == "ios-teethseg-full/v1" and info["asset_id"] == "a1" and info["split"] == "val"
    assert "npz_sha256" not in info and info["턱"] == "위턱" and info["색"] == "있음"


def test_lower_jaw_and_a_release_without_normals_or_colour(tmp_path: Path) -> None:
    make_cache(tmp_path / "s.npz", jaw_upper=False, normals=False)
    d = page_data(write_view(tmp_path / "s.npz", tmp_path / "s.html"))
    assert d["rgb"] is None
    nrm = decode(d["nrm"], "i1").reshape(-1, 3) / 127                 # 음영용으로 이웃 점에서 추정했다
    assert np.allclose(np.linalg.norm(nrm, axis=1), 1, atol=0.02)
    assert [r["label"] for r in d["legend"]["fdi"]][1:4] == ["31", "32", "41"]
    assert dict(d["info"])["normal"].startswith("없음")


def test_pca_normals_of_a_plane_point_along_its_axis() -> None:
    rng = np.random.default_rng(3)
    q = np.c_[rng.uniform(-5, 5, (300, 2)), np.zeros(300)]
    n = pca_normals(q, cKDTree(q))
    assert np.allclose(np.abs(n[:, 2]), 1, atol=1e-6)


def test_text_cannot_break_out_of_the_page(tmp_path: Path) -> None:
    make_cache(tmp_path / "s.npz")
    out = write_view(tmp_path / "s.npz", tmp_path / "s.html", {"asset_id": "</script><b>x"})
    text = out.read_text(encoding="utf-8")
    assert "</script><b>" not in text
    assert dict(page_data(out)["info"])["asset_id"] == "</script><b>x"


def test_release_views_and_pick(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    for name in ("A_1_upper", "B_2_upper", "C_3_upper"):
        make_cache(cache / f"{name}.npz")
    release = DownloadDataOutput(tmp_path, cache, ["A_1_upper", "B_2_upper"], ["C_3_upper"], [],
                                 {"dataset": "ios-teethseg", "version": "v1",
                                  "scans": {"C_3_upper": {"asset_id": "3", "split": "val"}}})
    assert pick(release, "all", None, 0) == ["A_1_upper", "B_2_upper", "C_3_upper"]
    assert pick(release, "train", None, 1) == ["A_1_upper"]
    assert pick(release, "val", ["B_2_upper"], 8) == ["B_2_upper"]       # 이름을 주면 split을 보지 않는다
    with pytest.raises(SystemExit, match="Z_9_upper"):
        pick(release, "all", ["Z_9_upper"], 8)
    [path] = write_views(release, ["C_3_upper"], tmp_path / "view")
    assert path == tmp_path / "view" / "C_3_upper.html"
    assert dict(page_data(path)["info"])["asset_id"] == "3"
