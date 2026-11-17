"""DDHStream: 캐시 없이 item마다 릴리스에서 받는 Dataset.  가짜 S3 (test_ddh_loader의 release)로 돌린다."""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import pytest
import torch

from dental_ai.dataset.DDHLoader import ddh_fetch
from dental_ai.dataset.DDHStream import DDHStreamDataset, UnusableScan
from dental_ai.dataset.algorithm.download_data.from_ddh import DDHError
from dental_ai.dataset.ToothSegLoader import ToothSegConfig, ToothSegDataset, tooth_seg_loader

from test_ddh_loader import NO_NORMALS, make_sample, release

WITH_NORMALS = ToothSegConfig(features=("xyz", "normal", "jaw"), train=False)


class Factory:
    """인자 없이 같은 가짜 버킷을 돌려주는 client 함수.  pickle되고, 몇 번 불렸는지 센다."""

    def __init__(self, s3) -> None:
        self.s3, self.calls = s3, 0

    def __call__(self):
        self.calls += 1
        return self.s3


def three_scans():
    return release({
        "a1": ("train", "p_ab12", make_sample(upper=True, normals=True)),
        "b2": ("train", "p_cd34", make_sample(upper=False, normals=True)),
        "c3": ("val", "p_ef56", make_sample(upper=True, normals=True)),
    })


def same_item(x, y) -> None:
    for field in ("coord", "grid", "features", "classes", "instances", "arch_positions", "offset_targets"):
        assert torch.equal(getattr(x, field), getattr(y, field)), field
    assert x.jaw_upper == y.jaw_upper


def test_stream_gives_the_same_items_as_the_downloaded_cache(tmp_path: Path) -> None:
    s3 = three_scans()
    cached = ddh_fetch("ios-teethseg", "v1", tmp_path, s3)
    disk = ToothSegDataset([cached.cache], cached.train, WITH_NORMALS)
    stream = DDHStreamDataset("ios-teethseg", "v1", "train", WITH_NORMALS, client=Factory(s3))
    assert len(stream) == len(disk) == 2
    by_asset = {n.split("_")[1]: n for n in disk.names}                 # DDH<pid>_<asset>_<jaw>
    for i, name in enumerate(stream.names):
        same_item(stream[i], disk[disk.names.index(by_asset[name.split("_")[1]])])


def test_stream_writes_nothing_and_downloads_on_every_item(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    s3 = three_scans()
    stream = DDHStreamDataset("ios-teethseg", "v1", None, NO_NORMALS, client=Factory(s3))
    assert sorted(n.split("_")[1] for n in stream.names) == ["a1", "b2", "c3"]
    for _ in range(2):
        for i in range(len(stream)):
            stream[i]
    npz_gets = [k for k in s3.gets if k.endswith(".npz")]
    assert len(npz_gets) == 6                                           # 2 epoch x 3 스캔, 매번 받는다
    assert os.listdir(tmp_path) == []


def test_memory_cache_keeps_recent_scans_in_ram_only(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    s3 = three_scans()
    stream = DDHStreamDataset("ios-teethseg", "v1", None, NO_NORMALS, client=Factory(s3), memory_cache=3)
    for _ in range(3):
        for i in range(len(stream)):
            stream[i]
    assert len([k for k in s3.gets if k.endswith(".npz")]) == 3
    assert os.listdir(tmp_path) == []


def test_splits_assets_and_exclude_pick_scans() -> None:
    s3 = three_scans()
    assert [n.split("_")[1] for n in DDHStreamDataset("ios-teethseg", "v1", "val", NO_NORMALS,
                                                      client=Factory(s3)).names] == ["c3"]
    picked = DDHStreamDataset("ios-teethseg", "v1", None, NO_NORMALS, client=Factory(s3), assets=["c3", "a1"])
    assert [n.split("_")[1] for n in picked.names] == ["c3", "a1"]
    left = DDHStreamDataset("ios-teethseg", "v1", ["train", "val"], NO_NORMALS, client=Factory(s3), exclude=["b2"])
    assert sorted(n.split("_")[1] for n in left.names) == ["a1", "c3"]
    with pytest.raises(DDHError):
        DDHStreamDataset("ios-teethseg", "v1", "dev", NO_NORMALS, client=Factory(s3))
    with pytest.raises(DDHError):
        DDHStreamDataset("ios-teethseg", "v1", "val", NO_NORMALS, client=Factory(s3), assets=["a1"])


def test_unusable_scan_is_replaced_and_reported(capsys) -> None:
    # 프렙(2)이 있는 스캔은 캐시 class가 없다: 학습은 멈추지 않고 다음 스캔을 받는다
    s3 = release({
        "a1": ("train", "p_ab12", make_sample(label_of_tooth=2)),
        "b2": ("train", "p_cd34", make_sample(upper=False)),
    })
    stream = DDHStreamDataset("ios-teethseg", "v1", "train", NO_NORMALS, client=Factory(s3))
    item = stream[stream.names.index(next(n for n in stream.names if "a1" in n))]
    assert "b2" in item.name and "a1" in next(iter(stream.skipped))
    assert "건너뛴다" in capsys.readouterr().err
    strict = DDHStreamDataset("ios-teethseg", "v1", "train", NO_NORMALS, client=Factory(s3), strict=True)
    with pytest.raises(UnusableScan):
        for i in range(len(strict)):
            strict[i]
    found = DDHStreamDataset("ios-teethseg", "v1", "train", NO_NORMALS, client=Factory(s3)).problems(workers=2)
    assert list(found) == [n for n in stream.names if "a1" in n]


def test_hash_mismatch_is_not_replaced() -> None:
    s3 = three_scans()
    key = next(k for k in s3.objects if k.endswith("a1.npz"))
    s3.objects[key] = s3.objects[key][:-1] + b"x"
    stream = DDHStreamDataset("ios-teethseg", "v1", "train", NO_NORMALS, client=Factory(s3))
    with pytest.raises(DDHError, match="해시"):
        stream[stream.names.index(next(n for n in stream.names if "a1" in n))]


def test_workers_get_their_own_client() -> None:
    factory = Factory(three_scans())
    stream = DDHStreamDataset("ios-teethseg", "v1", "train", NO_NORMALS, client=factory)
    assert factory.calls == 1                                            # manifest
    copy = pickle.loads(pickle.dumps(stream))
    assert copy._client is None and copy.names == stream.names
    copy[0]
    assert copy._client_factory.calls == 2                               # 받은 쪽 프로세스에서 새로 만든다
    batches = list(tooth_seg_loader(stream, batch_size=2, workers=2, shuffle=False))
    assert sum(len(b.names) for b in batches) == 2
