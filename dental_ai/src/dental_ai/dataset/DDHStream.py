"""DentalDataHub(DDH) 데이터셋 릴리스 (S3)를 캐시 없이 읽는 Dataset: item마다 S3에서 샘플 하나를 받는다.

    from dental_ai.dataset.DDHStream import DDHStreamDataset
    from dental_ai.dataset.ToothSegLoader import ToothSegConfig, tooth_seg_loader

    train = DDHStreamDataset("ios-teethseg-full", "v1", "train", ToothSegConfig())
    val = DDHStreamDataset("ios-teethseg-full", "v1", "val", ToothSegConfig(train=False))
    loader = tooth_seg_loader(train, batch_size=12, workers=12)

ddh-fetch (DDHLoader.py)는 릴리스 전체를 먼저 내려받아 data/ios/ddh_<데이터셋>_<버전>/cache에 둔다.  이 클래스는
디스크에 아무것도 쓰지 않는다:
- 만들 때 manifest.jsonl만 한 번 읽는다 (어떤 스캔이 어느 split인지, 각 npz의 sha256).
- __getitem__마다 그 스캔의 npz를 받아 sha256을 확인하고, from_ddh.sample_arrays로 캐시와 같은 배열(dtype까지)을
  메모리에서 만든다.  그 뒤는 ToothSegDataset과 같은 코드다 (augmentation, voxel downsample, feature).
  같은 릴리스 · 같은 seed면 캐시로 읽을 때와 같은 item이 나온다.
- epoch마다 다시 받는다.  스캔 하나가 수 MB이고 epoch마다 스캔 수만큼 받으므로, 학습 PC와 DDH가 같은 망에 있을
  때 맞는 방식이다.  memory_cache=N이면 worker마다 최근 N개 스캔을 메모리(RAM)에만 둔다 (기본 0: 두지 않는다).

잰 값 (2026-11-18, dev의 ios-teethseg-full/v1, 스캔 하나 약 4.9 MB, 학습 recipe 그대로):
- 스캔 하나 받기: Docker 내부망(DDH_S3_ENDPOINT=http://rustfs:9000) 4 ms, Windows Docker Desktop의
  host.docker.internal을 거치면 268 ms.  sha256 2 ms, np.load 1 ms, 캐시 배열로 바꾸기 7 ms.
- DataLoader (augmentation, voxel 포함): 내부망에서 worker 6개 초당 20.8개, 12개 28.3개.  host.docker.internal로는
  worker 6개 초당 6.3개.  ToothSegNet instance 학습(RTX 5080)이 쓰는 속도는 초당 약 15.6개다.
  같은 PC에서 학습하면 학습 컨테이너를 dental-data-hub_ddh-net에 붙이고 rustfs:9000으로 받는다.

접속 정보는 ddh-fetch와 같은 환경 변수다 (DDH의 trainer 키, ddh-datasets 읽기 전용):

    DDH_S3_ENDPOINT          http://<서버>:9000
    DDH_TRAINER_ACCESS_KEY
    DDH_TRAINER_SECRET_KEY

DataLoader worker: S3 클라이언트는 프로세스마다 처음 쓸 때 만든다 (boto3 클라이언트는 프로세스 사이에 나눠 쓸 수
없다).  Dataset을 worker로 보낼 때(pickle) 클라이언트와 메모리 캐시는 빼고 보낸다.

쓸 수 없는 스캔: from_ddh가 건너뛰는 스캔(label이 0, 1 밖 — -1 미분류, 2 Prep-Tooth, 3 Scanbody, 4 Peek —, 또는
턱을 알 수 없음)은 manifest만으로는 모르고 받아 봐야 안다.  학습이 도중에 멈추지 않게, 그런 스캔이 걸리면
경고하고 다음 스캔으로 대신한다.  strict=True면 DDHError를 낸다.  미리 다 확인하려면 problems()를 부른다
(전체를 한 번 받아 보고, 아무것도 남기지 않는다).

이름은 DDH<환자>_<asset>이다 (patient_split이 _ 앞을 환자로 본다).  ddh-fetch 캐시 이름과 달리 턱(_upper/_lower)이
없다: 턱은 npz를 받아야 알 수 있다.  item.jaw_upper에 들어 있다.
"""
from __future__ import annotations

import os
import sys
from collections import OrderedDict
from collections.abc import Callable, Iterable

import numpy as np

from .algorithm.download_data.from_ddh import (SPLITS, DDHError, ddh_client, fetch_sample, read_manifest,
                                               sample_arrays, sample_problem)
from .algorithm.augmentation.augmentation import ScanSample
from .ToothSegLoader import ToothSegConfig, ToothSegDataset, ToothSegItem


class UnusableScan(DDHError):
    """from_ddh가 건너뛰는 스캔 (label이 0, 1 밖이거나 턱을 알 수 없음).  해시가 다른 것과 달리 대신할 수 있다."""


def stream_name(entry: dict) -> str:
    """DDH<환자>_<asset>.  파일 이름이 되지는 않지만 ddh-fetch와 같은 규칙으로 검사한다."""
    pid, asset = entry["patient_pid"].replace("_", ""), entry["asset_id"]
    if not (pid.isalnum() and asset.isalnum()):
        raise DDHError(f"manifest의 ID가 이상하다: {pid!r}, {asset!r}")
    return f"DDH{pid}_{asset}"


class DDHStreamDataset(ToothSegDataset):
    """ToothSegDataset과 같은 item을 내지만, 캐시 대신 DDH 릴리스에서 item마다 받는다."""

    def __init__(self, dataset: str, version: str, splits: str | Iterable[str] | None = "train",
                 config: ToothSegConfig | None = None, *, assets: Iterable[str] | None = None,
                 exclude: Iterable[str] = (), client: Callable[[], object] = ddh_client, memory_cache: int = 0,
                 strict: bool = False) -> None:
        """splits: "train" / ["train", "val"] / None(전부).  assets를 주면 그 asset_id만 (순서도 그대로).
        exclude: 뺄 asset_id.  client: 인자 없이 boto3 S3 클라이언트를 돌려주는 함수 (프로세스마다 부른다)."""
        super().__init__([], [], config)
        self.dataset, self.version = dataset, version
        self._client_factory = client
        self._client, self._client_pid = None, None
        self._memory_cache = max(0, int(memory_cache))
        self._lru: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()
        self.strict = strict
        self.skipped: dict[str, str] = {}          # 이 프로세스에서 걸린 쓸 수 없는 스캔: 이름 -> 이유

        entries = read_manifest(self._cli(), dataset, version)
        if splits is not None:
            want = {splits} if isinstance(splits, str) else set(splits)
            unknown = sorted(want - set(SPLITS))
            if unknown:
                raise DDHError(f"split {unknown}: {list(SPLITS)} 중에서 고른다")
            entries = [e for e in entries if e["split"] in want]
        drop = set(exclude)
        if assets is not None:
            by_asset = {e["asset_id"]: e for e in entries}
            missing = [a for a in assets if a not in by_asset]
            if missing:
                raise DDHError(f"{dataset}/{version}에 없는 asset (또는 다른 split): {missing[:5]}")
            entries = [by_asset[a] for a in assets]
        entries = [e for e in entries if e["asset_id"] not in drop]
        self.entries: dict[str, dict] = {stream_name(e): e for e in entries}
        self.names = list(self.entries)
        self.missing = []

    # ------------------------------------------------------------- S3
    def _cli(self):
        if self._client is None or self._client_pid != os.getpid():
            self._client, self._client_pid = self._client_factory(), os.getpid()
        return self._client

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state["_client"], state["_client_pid"] = None, None
        state["_lru"] = OrderedDict()
        return state

    def arrays(self, name: str) -> dict[str, np.ndarray]:
        """스캔 하나 -> 캐시 배열 (메모리).  쓸 수 없는 스캔이면 DDHError."""
        cached = self._lru.get(name)
        if cached is not None:
            self._lru.move_to_end(name)
            return cached
        sample = fetch_sample(self._cli(), self.dataset, self.version, self.entries[name])
        problem = sample_problem(sample)
        if problem:
            raise UnusableScan(f"{name} ({self.entries[name]['asset_id']}): {problem}")
        arrays = sample_arrays(sample)
        if self._memory_cache:
            self._lru[name] = arrays
            while len(self._lru) > self._memory_cache:
                self._lru.popitem(last=False)
        return arrays

    def load(self, name: str) -> tuple[ScanSample, bool]:
        return self.scan_from_arrays(name, self.arrays(name))

    # ---------------------------------------------------------- items
    def __getitem__(self, i: int) -> ToothSegItem:
        for k in range(len(self.names)):
            j = (i + k) % len(self.names)
            name = self.names[j]
            if name in self.skipped:
                continue
            try:
                return super().__getitem__(j)
            except UnusableScan as e:                   # 해시가 다르면(DDHError) 릴리스가 깨진 것: 대신하지 않는다
                if self.strict:
                    raise
                self.skipped[name] = str(e)
                print(f"DDHStream: 건너뛴다 — {e}", file=sys.stderr, flush=True)
        raise DDHError(f"{self.dataset}/{self.version}: 쓸 수 있는 스캔이 없다")

    def problems(self, workers: int = 8) -> dict[str, str]:
        """모든 스캔을 한 번 받아 쓸 수 없는 것만 돌려준다 (이름 -> 이유).  받은 것은 남기지 않는다.
        찾은 스캔은 skipped에 넣으므로, DataLoader를 만들기 전에 부르면 worker도 처음부터 건너뛴다."""
        import threading
        from concurrent.futures import ThreadPoolExecutor

        local = threading.local()                          # 스레드마다 클라이언트 하나

        def check(name: str) -> tuple[str, str | None]:
            if getattr(local, "client", None) is None:
                local.client = self._client_factory()
            sample = fetch_sample(local.client, self.dataset, self.version, self.entries[name])
            return name, sample_problem(sample)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            found = {n: why for n, why in pool.map(check, self.names) if why}
        self.skipped.update(found)
        return found


def main() -> None:
    """처리량 확인: python -m dental_ai.dataset.DDHStream ios-teethseg-full v1 --items 300 --workers 6"""
    import argparse
    import time

    from .ToothSegLoader import tooth_seg_loader

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("dataset")
    ap.add_argument("version")
    ap.add_argument("--split", default="train")
    ap.add_argument("--items", type=int, default=300, help="이만큼 받고 멈춘다")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--problems", action="store_true", help="먼저 전체를 한 번 받아 쓸 수 없는 스캔을 센다")
    args = ap.parse_args()
    t0 = time.time()
    ds = DDHStreamDataset(args.dataset, args.version, args.split, ToothSegConfig())
    print(f"{args.dataset}/{args.version} {args.split}: {len(ds)} scans, manifest {time.time() - t0:.1f}s")
    if args.problems:
        t0 = time.time()
        found = ds.problems()
        print(f"problems(): {len(found)} unusable of {len(ds)} in {time.time() - t0:.0f}s")
        for name, why in list(found.items())[:10]:
            print("  ", name, why)
    t0, n, points = time.time(), 0, 0
    for batch in tooth_seg_loader(ds, batch_size=args.batch, workers=args.workers):
        n += len(batch.names)
        points += int(batch.coord.shape[0])
        if n >= args.items:
            break
    dt = time.time() - t0
    print(f"{n} items in {dt:.1f}s = {n / dt:.1f} items/s ({points / max(n, 1):.0f} points per item after voxel)")


if __name__ == "__main__":
    main()
