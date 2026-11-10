"""DentalDataHub(DDH) 데이터셋 릴리스 (S3) -> ToothSegLoader가 읽는 npz 캐시.

    dental_ai ddh-fetch ios-teethseg-full v1            # -> data/ios/ddh_ios-teethseg-full_v1/
    (= python -m dental_ai.dataset.DDHLoader <인자>)

받는 일은 algorithm/download_data/from_ddh.py가 한다.  폴더 구조, meta 내용, 샘플 -> 캐시 규칙도 거기 있다.
여기에는 이 명령과, from_ddh를 한 줄로 부르는 ddh_fetch만 있다.

학습에는 ios-teethseg-full을 쓴다 (전체 정점, normal, jaw).  접속 정보는 환경 변수로 준다 (DDH의 trainer 키):

    DDH_S3_ENDPOINT          http://<서버>:9000
    DDH_TRAINER_ACCESS_KEY
    DDH_TRAINER_SECRET_KEY

    from dental_ai.dataset.DDHLoader import ddh_fetch
    from dental_ai.dataset.ToothSegLoader import ToothSegConfig, ToothSegDataset, tooth_seg_loader

    ddh = ddh_fetch("ios-teethseg-full", "v1")  # 다 받았으면 S3에 묻지 않는다
    train = ToothSegDataset([ddh.cache], ddh.train, ToothSegConfig())
    val = ToothSegDataset([ddh.cache], ddh.val, ToothSegConfig(train=False))
    loader = tooth_seg_loader(train, batch_size=12, workers=12)
    ddh.meta["definition"]["preprocess"]        # 릴리스를 만든 설정 (dataset.toml)
    ddh.meta["scans"][ddh.train[0]]["asset_id"]  # 캐시 이름 -> DDH의 asset_id, patient_pid, label_version
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .algorithm.download_data.download_data import DownloadDataInput, DownloadDataOutput
from .algorithm.download_data.from_ddh import DDHError, from_ddh


def ddh_fetch(dataset: str, version: str, out: Path | str = "data/ios", client=None) -> DownloadDataOutput:
    """from_ddh를 한 줄로.  받지 못하면 DDHError를 낸다 (이유는 from_ddh가 stderr에 먼저 쓴다)."""
    release = DownloadDataOutput()
    if not from_ddh(DownloadDataInput(dataset, version, Path(out), client), release):
        raise DDHError(f"{dataset}/{version}: 받지 못했다")
    return release


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help="DDH 데이터셋 이름 (예: ios-teethseg-full)")
    ap.add_argument("version", help="릴리스 버전 (예: v1)")
    ap.add_argument("--out", type=Path, default=Path("data/ios"), help="이 아래 ddh_<dataset>_<version>/을 만든다")
    args = ap.parse_args()
    release = DownloadDataOutput()
    if not from_ddh(DownloadDataInput(args.dataset, args.version, args.out), release):
        raise SystemExit(1)                     # 이유는 from_ddh가 stderr에 썼다
    print(release.root)


if __name__ == "__main__":
    main()
