"""from_ddh / DDHLoader: S3 없이 가짜 클라이언트로 릴리스를 받아 ToothSegDataset까지 돌린다."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
from botocore.exceptions import ClientError

from dental_ai.dataset.DDHLoader import DDHError, ddh_fetch
from dental_ai.dataset.algorithm.download_data.download_data import DownloadDataInput, DownloadDataOutput
from dental_ai.dataset.algorithm.download_data.from_ddh import from_ddh, jaw_from_fdi
from dental_ai.dataset.ToothSegLoader import ToothSegConfig, ToothSegDataset, patient_split, read_split, tooth_seg_loader

ROOT = "ios-teethseg/v1/"
NO_NORMALS = ToothSegConfig(features=("xyz", "jaw"), train=False)


def make_sample(upper: bool = True, label_of_tooth: int = 1, normals: bool = False, numbered: bool = True) -> dict:
    """치은 + 치아 2개, 점 1000개."""
    n = 1000
    rng = np.random.default_rng(0)
    points = rng.uniform(-1.0, 1.0, (n, 3)).astype(np.float32)
    label, fdi, instance = np.zeros(n, np.int32), np.zeros(n, np.int32), np.zeros(n, np.int32)
    first, second = (11, 21) if upper else (31, 41)
    label[:200], fdi[:200], instance[:200] = label_of_tooth, first if numbered else 0, 5
    label[200:400], fdi[200:400], instance[200:400] = label_of_tooth, second if numbered else 0, 3
    sample = dict(points=points, vertex_index=np.arange(n, dtype=np.int32), center=np.array([10.0, -20.0, 5.0]),
                  scale=np.float64(40.0), fdi=fdi, label=label, instance=instance)
    if normals:
        sample["normals"] = np.tile(np.array([0.0, 0.0, 1.0], np.float32), (n, 1))
    return sample


def npz_bytes(sample: dict) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **sample)
    return buf.getvalue()


class FakeS3:
    """get_object / list_objects_v2만 있는 ddh-datasets 버킷."""

    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects
        self.gets: list[str] = []

    def get_object(self, Bucket: str, Key: str) -> dict:
        self.gets.append(Key)
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey", "Message": Key}}, "GetObject")
        return {"Body": io.BytesIO(self.objects[Key])}

    def list_objects_v2(self, Bucket: str, Prefix: str, Delimiter: str) -> dict:
        below = {k[len(Prefix):].split("/")[0] for k in self.objects if k.startswith(Prefix) and "/" in k[len(Prefix):]}
        return {"CommonPrefixes": [{"Prefix": f"{Prefix}{d}/"} for d in below]}


def release(samples: dict[str, tuple[str, str, dict]]) -> FakeS3:
    """{asset_id: (split, patient_pid, sample)} -> 릴리스 객체들."""
    objects = {ROOT + "dataset.toml": b"name = 'ios-teethseg'\n", ROOT + "README.md": b"# card\n"}
    lines = []
    for asset, (split, pid, sample) in samples.items():
        data = npz_bytes(sample)
        objects[f"{ROOT}{split}/{asset}.npz"] = data
        # label_version, label_sha256은 일부러 뺀다: v1처럼 그 값이 생기기 전의 manifest도 받아야 한다
        lines.append(json.dumps(dict(asset_id=asset, split=split, patient_pid=pid, npz_path=f"{split}/{asset}.npz",
                                     npz_sha256=hashlib.sha256(data).hexdigest(), n_points=len(sample["points"]))))
    objects[ROOT + "manifest.jsonl"] = ("\n".join(lines) + "\n").encode()
    return FakeS3(objects)


def test_sample_becomes_the_cache_format(tmp_path: Path) -> None:
    sample = make_sample(upper=False)
    ddh = ddh_fetch("ios-teethseg", "v1", tmp_path, release({"a1": ("train", "p_ab12", sample)}))

    root = tmp_path / "ddh_ios-teethseg_v1"
    name = "DDHpab12_a1_lower"
    assert (ddh.root, ddh.cache) == (root, root / "cache")
    assert (ddh.train, ddh.val, ddh.test) == ([name], [], [])
    assert read_split(root / "splits/train.txt") == [name]
    with np.load(ddh.cache / f"{name}.npz") as d:
        assert not d["jaw"]
        assert d["xyz"].dtype == np.float32
        np.testing.assert_allclose(d["xyz"], sample["points"].astype(np.float64) * 40.0 + [10.0, -20.0, 5.0], atol=1e-4)
        assert d["cls"][:200].tolist() == [1] * 200         # 31 -> 첫 사분면의 1번
        assert d["cls"][200:400].tolist() == [9] * 200      # 41 -> 둘째 사분면의 1번
        assert d["cls"][400:].max() == 0
        assert d["ins"][:200].tolist() == [1] * 200         # 다수결 class 순서로 1..K
        assert d["ins"][200:400].tolist() == [2] * 200
    assert (root / "dataset.toml").exists() and (root / "manifest.jsonl").exists()


def test_release_needs_no_folder_layout(tmp_path: Path) -> None:
    s3 = release({"a1": ("train", "p_ab12", make_sample()), "a2": ("val", "p_cd34", make_sample(upper=False)),
                  "a3": ("test", "p_ef56", make_sample(label_of_tooth=2))})
    ddh = ddh_fetch("ios-teethseg", "v1", tmp_path, s3)
    assert (ddh.train, ddh.val, ddh.test) == (["DDHpab12_a1_upper"], ["DDHpcd34_a2_lower"], [])
    assert ToothSegDataset([ddh.cache], ddh.train + ddh.val, NO_NORMALS).missing == []
    assert (ddh.meta["dataset"], ddh.meta["version"]) == ("ios-teethseg", "v1")
    assert ddh.meta["definition"] == {"name": "ios-teethseg"}
    scan = ddh.meta["scans"]["DDHpcd34_a2_lower"]
    assert (scan["asset_id"], scan["patient_pid"], scan["split"]) == ("a2", "p_cd34", "val")
    assert [(s["asset_id"], s["split"]) for s in ddh.meta["skipped"]] == [("a3", "test")]


def test_cache_feeds_the_existing_loader(tmp_path: Path) -> None:
    s3 = release({"a1": ("train", "p_ab12", make_sample()), "a2": ("train", "p_cd34", make_sample(upper=False))})
    ddh = ddh_fetch("ios-teethseg", "v1", tmp_path, s3)
    names = ddh.train
    dataset = ToothSegDataset([ddh.cache], names, NO_NORMALS)
    assert dataset.missing == []
    batch = next(iter(tooth_seg_loader(dataset, batch_size=2, workers=0, pin_memory=False)))
    assert batch.features.shape[1] == NO_NORMALS.in_ch
    assert sorted(batch.jaw_upper.tolist()) == [False, True]
    # patient_split은 이름의 첫 _ 앞을 환자로 본다
    train, val = patient_split(names + ["DDHpab12_a9_lower"], val_frac=0.34)
    assert sum(n.startswith("DDHpab12") for n in val) in (0, 2)


def test_missing_normals_are_refused_where_they_matter(tmp_path: Path) -> None:
    s3 = release({"a1": ("train", "p_ab12", make_sample()), "a2": ("train", "p_cd34", make_sample(normals=True))})
    cache = ddh_fetch("ios-teethseg", "v1", tmp_path, s3).cache
    no_normals, with_normals = "DDHpab12_a1_upper", "DDHpcd34_a2_upper"
    for features in (("xyz", "normal"), ("xyz", "curvature")):
        cfg = ToothSegConfig(features=features, train=False)
        dataset = ToothSegDataset([cache], [no_normals, with_normals], cfg)
        with pytest.raises(ValueError, match="normal이 없는"):
            dataset[0]
        assert dataset[1].features.shape[1] == cfg.in_ch
    assert ToothSegDataset([cache], [no_normals], NO_NORMALS)[0].features.shape[1] == NO_NORMALS.in_ch


@pytest.mark.parametrize("bad", [make_sample(label_of_tooth=2), make_sample(numbered=False)])
def test_scans_without_a_cache_class_or_jaw_are_skipped(tmp_path: Path, bad: dict, capsys) -> None:
    s3 = release({"good": ("train", "p_ab12", make_sample()), "bad": ("train", "p_cd34", bad)})
    ddh = ddh_fetch("ios-teethseg", "v1", tmp_path, s3)
    assert ddh.train == ["DDHpab12_good_upper"]
    assert len(list(ddh.cache.glob("*.npz"))) == 1
    assert "skipped bad" in capsys.readouterr().out
    assert [s["asset_id"] for s in ddh.meta["skipped"]] == ["bad"]


def test_jaw_field_wins_over_tooth_numbers(tmp_path: Path) -> None:
    unnumbered = make_sample(numbered=False)                # 번호 없는 치아: jaw 필드가 있으면 받는다
    unnumbered["jaw"] = np.uint8(0)
    bite = make_sample()
    bite["jaw"] = np.uint8(2)
    s3 = release({"a1": ("train", "p_ab12", unnumbered), "a2": ("train", "p_cd34", bite)})
    ddh = ddh_fetch("ios-teethseg", "v1", tmp_path, s3)
    assert ddh.train == ["DDHpab12_a1_lower"]
    with np.load(ddh.cache / "DDHpab12_a1_lower.npz") as d:
        assert not d["jaw"]
        assert d["cls"][:400].tolist() == [17] * 400        # 위치 모르는 치아
        assert d["ins"][:200].tolist() == [2] * 200         # 치아는 그대로 둘이다.  class가 같으면 원래 id 순서
        assert d["ins"][200:400].tolist() == [1] * 200


def test_jaw_from_fdi() -> None:
    assert jaw_from_fdi(np.array([0, 11, 28])) is True
    assert jaw_from_fdi(np.array([0, 31, 48])) is False
    assert jaw_from_fdi(np.array([0, 55])) is True          # 유치
    assert jaw_from_fdi(np.array([0, 75])) is False
    assert jaw_from_fdi(np.array([0, 0])) is None
    assert jaw_from_fdi(np.array([11, 31])) is None


def test_hash_mismatch_stops_and_the_rerun_resumes(tmp_path: Path, capsys) -> None:
    s3 = release({"a1": ("train", "p_ab12", make_sample()), "a2": ("val", "p_cd34", make_sample())})
    good = s3.objects[ROOT + "val/a2.npz"]
    s3.objects[ROOT + "val/a2.npz"] = good[:-1] + bytes([good[-1] ^ 1])
    with pytest.raises(DDHError):
        ddh_fetch("ios-teethseg", "v1", tmp_path, s3)
    assert "val/a2.npz: 해시" in capsys.readouterr().err
    root = tmp_path / "ddh_ios-teethseg_v1"
    assert not (root / "meta.json").exists()                # 다 받은 것으로 보이면 안 된다

    s3.objects[ROOT + "val/a2.npz"] = good
    s3.gets.clear()
    ddh = ddh_fetch("ios-teethseg", "v1", tmp_path, s3)
    assert ROOT + "train/a1.npz" not in s3.gets             # 이미 받은 스캔은 다시 받지 않는다
    assert (ddh.train, ddh.val) == (["DDHpab12_a1_upper"], ["DDHpcd34_a2_upper"])
    assert sorted(ddh.meta["scans"]) == ["DDHpab12_a1_upper", "DDHpcd34_a2_upper"]   # 이어 받은 스캔도 meta에 있다


def test_finished_release_needs_no_s3(tmp_path: Path) -> None:
    s3 = release({"a1": ("train", "p_ab12", make_sample())})
    first = ddh_fetch("ios-teethseg", "v1", tmp_path, s3)
    s3.gets.clear()
    assert ddh_fetch("ios-teethseg", "v1", tmp_path, s3) == first
    assert s3.gets == []


def test_missing_release_names_what_exists(tmp_path: Path, capsys) -> None:
    s3 = release({"a1": ("train", "p_ab12", make_sample())})
    with pytest.raises(DDHError):
        ddh_fetch("ios-teethseg", "v9", tmp_path, s3)
    assert "'v1'" in capsys.readouterr().err
    assert not (tmp_path / "ddh_ios-teethseg_v9").exists()


@pytest.mark.parametrize("dataset, version, why", [
    ("ios-teethseg", "v9", "릴리스가 없다"),
    ("../ios-teethseg", "v1", "영문 소문자"),           # 폴더 이름이 된다
    ("ios-teethseg", "v1/..", "v<번호>"),
])
def test_from_ddh_says_false_and_leaves_out_empty(tmp_path: Path, capsys, dataset: str, version: str, why: str) -> None:
    s3 = release({"a1": ("train", "p_ab12", make_sample())})
    out = DownloadDataOutput()
    assert not from_ddh(DownloadDataInput(dataset, version, tmp_path, s3), out)
    assert out == DownloadDataOutput()
    assert why in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_from_ddh_without_keys_is_false(tmp_path: Path, capsys, monkeypatch) -> None:
    for name in ("DDH_S3_ENDPOINT", "DDH_TRAINER_ACCESS_KEY", "DDH_TRAINER_SECRET_KEY"):
        monkeypatch.delenv(name, raising=False)
    assert not from_ddh(DownloadDataInput("ios-teethseg", "v1", tmp_path), DownloadDataOutput())
    assert "DDH_TRAINER_SECRET_KEY" in capsys.readouterr().err
