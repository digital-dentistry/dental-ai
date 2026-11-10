"""DentalDataHub(DDH) 데이터셋 릴리스 (S3) -> ToothSegLoader가 읽는 npz 캐시.

학습에는 ios-teethseg-full을 쓴다: 전체 정점, 면에서 구한 normal, jaw가 있어서 preprocess가 만든 캐시와 같은
값이 나온다.  ios-teethseg는 스캔마다 16,000점이고 v1에는 normal과 jaw가 없다.

접속 정보는 환경 변수로 준다.  키는 DDH의 trainer 키 (ddh-datasets 읽기 전용)다:

    DDH_S3_ENDPOINT          http://<서버>:9000
    DDH_TRAINER_ACCESS_KEY
    DDH_TRAINER_SECRET_KEY

<out_dir>/ddh_<데이터셋>_<버전>/                  (DownloadDataOutput.root)
    cache/<이름>.npz         ToothSegPreProcessing과 같은 형식 (DownloadDataOutput.cache)
    splits/<split>.txt       train / val / test.  DDH가 정한 환자 단위 split 그대로다
    manifest.jsonl           릴리스에서 받은 그대로
    dataset.toml, README.md  릴리스에서 받은 그대로
    meta.json                DownloadDataOutput.meta.  이 파일이 있으면 다 받은 것이다 (맨 마지막에 쓴다)

DownloadDataOutput.meta (= meta.json):

    dataset, version  받은 릴리스
    definition        릴리스를 만든 dataset.toml (description, select, preprocess)
    scans             캐시 이름 -> manifest 한 줄 (asset_id, patient_pid, split, label_version, ...)
    skipped           캐시로 바꾸지 못한 스캔 [{asset_id, patient_pid, split, reason}]

릴리스는 바뀌지 않으므로 다 받았으면 S3에 다시 묻지 않는다 (키도 필요 없다).  도중에 끊기면 다시 불러 이어
받는다.  이름은 <환자>_<asset>_<jaw>라서 patient_split과 jaw_of가 그대로 돈다.

    ddh = DownloadDataOutput()
    if not from_ddh(DownloadDataInput("ios-teethseg-full", "v1"), ddh):
        raise SystemExit(1)                     # 이유는 stderr에 나왔다
    train = ToothSegDataset([ddh.cache], ddh.train, ToothSegConfig())

한 줄로 부르려면 dental_ai.dataset.DDHLoader.ddh_fetch (받지 못하면 DDHError).

릴리스 샘플 -> 캐시:

    xyz       points * scale + center (mm)
    jaw       샘플의 jaw (0 lower, 1 upper, 2 bite).  bite는 건너뛴다.  jaw가 없는 릴리스는 FDI 사분면으로
              정하고, 치아 번호가 없거나 두 턱이 섞인 스캔은 건너뛴다
    cls, ins  ToothSegPreProcessing과 같은 규칙.  label이 0 (치은), 1 (치아) 밖인 스캔은 건너뛴다
              (치은으로 섞지 않는다.  -1 미분류, 2 Prep-Tooth, 3 Scanbody, 4 Peek는 cache class가 없다.
              VertexLabel의 2는 스캔바디라 DDH의 2와 뜻이 다르다)
    nrm       normals.  없으면 0으로 채우고, normal을 쓰는 feature는 ToothSegDataset이 거부한다
    rgb       colors가 있을 때만
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import sys
import time
import tomllib
from pathlib import Path

import boto3
import numpy as np
import torch
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from ...ToothSegPreProcessing import fdi_to_class, ordered_instances, real_colour, save_cache
from ..augmentation.augmentation import UNKNOWN_TOOTH
from .download_data import DownloadDataInput, DownloadDataOutput

BUCKET = "ddh-datasets"
SPLITS = ("train", "val", "test")
DATASET_NAME = re.compile(r"[a-z0-9][a-z0-9-]*")    # 둘 다 폴더 이름이 된다
VERSION_NAME = re.compile(r"v\d+")

DDH_GINGIVA, DDH_TOOTH = 0, 1        # DDH docs/DataFormat.md의 label
DDH_LOWER, DDH_UPPER = 0, 1          # 샘플의 jaw.  2는 bite
UPPER_QUADRANTS = (1, 2, 5, 6)       # FDI 사분면: 영구치 1..4, 유치 5..8
LOWER_QUADRANTS = (3, 4, 7, 8)


class DDHError(Exception):
    """릴리스를 받을 수 없다.  from_ddh는 이것을 False로 바꾼다 (ddh_client, DDHLoader.ddh_fetch는 그대로 낸다)."""


def from_ddh(inp: DownloadDataInput, out: DownloadDataOutput) -> bool:
    """릴리스 하나를 <out_dir>/ddh_<dataset>_<version>/로 받아 out을 채운다.  이미 다 받았으면 S3에 묻지 않는다.

    False: 받지 못했다 (이름이 틀림, 키가 없음, 릴리스가 없음, 해시가 다름, S3 오류).  이유는 stderr에 쓴다.
    다 받은 스캔은 남으므로 다시 부르면 이어 받는다.
    """
    if not (DATASET_NAME.fullmatch(inp.dataset) and VERSION_NAME.fullmatch(inp.version)):
        return _refuse(f"{inp.dataset} {inp.version}: 데이터셋은 영문 소문자·숫자·-, 버전은 v<번호>다")
    root = Path(inp.out_dir) / f"ddh_{inp.dataset}_{inp.version}"
    if not (root / "meta.json").exists():
        try:
            _download(inp.client or ddh_client(), inp.dataset, inp.version, root)
        except (DDHError, ClientError, BotoCoreError) as e:
            return _refuse(f"{inp.dataset}/{inp.version}: {e}")
    _load(root, out)
    return True


def ddh_client(endpoint: str | None = None, access_key: str | None = None, secret_key: str | None = None):
    """boto3 S3 client.  주지 않은 값은 환경 변수에서 읽는다 (맨 위 설명)."""
    env = {
        "DDH_S3_ENDPOINT": endpoint,
        "DDH_TRAINER_ACCESS_KEY": access_key,
        "DDH_TRAINER_SECRET_KEY": secret_key,
    }
    env = {
        name: value or os.environ.get(name) for name, value in env.items()
    }
    missing = [name for name, value in env.items() if not value]
    if missing:
        raise DDHError(f"환경 변수가 없다: {', '.join(missing)}")
    return boto3.client(
        "s3",
        endpoint_url=env["DDH_S3_ENDPOINT"],
        aws_access_key_id=env["DDH_TRAINER_ACCESS_KEY"],
        aws_secret_access_key=env["DDH_TRAINER_SECRET_KEY"],
        region_name="us-east-1",
        config=Config(
            s3={"addressing_style": "path"},
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        ),
    )


# ---------------------------------------------------------------- 샘플 -> 캐시
def jaw_from_fdi(fdi: np.ndarray) -> bool | None:
    """치아 번호 -> 위턱인가.  번호가 없거나 두 턱이 섞여 있으면 None."""
    quadrants = fdi[fdi > 0] // 10
    upper, lower = bool(np.isin(quadrants, UPPER_QUADRANTS).any()), bool(np.isin(quadrants, LOWER_QUADRANTS).any())
    return None if upper == lower else upper


def sample_jaw(sample: dict[str, np.ndarray]) -> bool | None:
    """샘플이 위턱인가.  bite이거나 턱을 알 수 없으면 None."""
    if "jaw" not in sample:
        return jaw_from_fdi(sample["fdi"])
    jaw = int(sample["jaw"])
    return jaw == DDH_UPPER if jaw in (DDH_LOWER, DDH_UPPER) else None


def sample_problem(sample: dict[str, np.ndarray]) -> str | None:
    """이 샘플을 캐시로 바꿀 수 없는 이유."""
    other = sorted(set(np.unique(sample["label"]).tolist()) - {DDH_GINGIVA, DDH_TOOTH})
    if other:
        return f"label {other} has no cache class yet"
    if sample_jaw(sample) is None:
        return "no jaw: a bite scan, or no jaw field and the tooth numbers don't tell"
    return None


def save_sample(sample: dict[str, np.ndarray], path: Path) -> None:
    """sample_problem이 None인 샘플 -> npz.  도중에 끊겨도 깨진 캐시가 남지 않게 다 쓴 뒤 옮긴다."""
    jaw_upper = sample_jaw(sample)
    xyz = sample["points"].astype(np.float64) * sample["scale"] + sample["center"]
    tooth = sample["label"] == DDH_TOOTH
    cls = fdi_to_class(np.where(tooth, sample["fdi"], 0), jaw_upper)
    cls[tooth & (cls == 0)] = UNKNOWN_TOOTH
    ins = ordered_instances(sample["instance"], cls)
    normals = torch.from_numpy(sample["normals"]) if "normals" in sample else torch.zeros(len(xyz), 3)
    rgb = real_colour(torch.from_numpy(sample["colors"].astype(np.float32) / 255.0)) if "colors" in sample else None
    part = path.with_suffix(".part.npz")
    save_cache(part, torch.from_numpy(xyz.astype(np.float32)), normals, cls, ins, jaw_upper, rgb)
    os.replace(part, path)


def scan_name(entry: dict, jaw_upper: bool) -> str:
    """<환자>_<asset>_<jaw>.  같은 환자가 같은 턱을 다시 스캔해도 겹치지 않고, 다른 데이터셋과도 겹치지 않는다."""
    pid, asset = entry["patient_pid"].replace("_", ""), entry["asset_id"]   # p_<hex>: patient_split이 _로 나눈다
    if not (pid.isalnum() and asset.isalnum()):      # 파일 이름이 된다
        raise DDHError(f"manifest의 ID가 이상하다: {pid!r}, {asset!r}")
    return f"DDH{pid}_{asset}_{'upper' if jaw_upper else 'lower'}"


# ------------------------------------------------------------------- 받기
def _refuse(reason: str) -> bool:
    print(f"from_ddh: {reason}", file=sys.stderr, flush=True)
    return False


def _get(cli, key: str) -> bytes:
    return cli.get_object(Bucket=BUCKET, Key=key)["Body"].read()


def _children(cli, prefix: str) -> list[str]:
    """prefix 바로 아래 폴더 이름."""
    listing = cli.list_objects_v2(Bucket=BUCKET, Prefix=prefix, Delimiter="/")
    return sorted(p["Prefix"][len(prefix):].rstrip("/") for p in listing.get("CommonPrefixes", []))


def _get_manifest(cli, dataset: str, version: str) -> bytes:
    try:
        return _get(cli, f"{dataset}/{version}/manifest.jsonl")
    except ClientError as e:
        if e.response["Error"]["Code"] not in ("404", "NoSuchKey", "NotFound"):
            raise
    # manifest는 DDH가 맨 마지막에 올린다: 폴더가 있어도 manifest가 없으면 만들다 실패한 버전이다
    raise DDHError(f"릴리스가 없다.  있는 데이터셋 {_children(cli, '')}, {dataset}의 버전 {_children(cli, f'{dataset}/')}")


def _download(cli, dataset: str, version: str, root: Path) -> None:
    """릴리스 -> root.  meta.json을 맨 마지막에 쓴다."""
    manifest = _get_manifest(cli, dataset, version)
    entries = [json.loads(line) for line in manifest.decode().splitlines() if line]
    cache = root / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    # 끊긴 뒤 이어 받기: 다 쓴 캐시만 이 이름을 갖는다
    done = {p.stem.split("_")[1]: p.stem for p in cache.glob("DDH*_*_*.npz") if ".part" not in p.stem}

    names: dict[str, list[str]] = {split: [] for split in SPLITS}
    scans: dict[str, dict] = {}
    skipped: list[dict] = []
    no_normals = 0
    t0 = time.time()
    for k, entry in enumerate(entries, 1):
        asset = entry["asset_id"]
        name = done.get(asset)
        if name is None:
            data = _get(cli, f"{dataset}/{version}/{entry['npz_path']}")
            if hashlib.sha256(data).hexdigest() != entry["npz_sha256"]:
                raise DDHError(f"{entry['npz_path']}: 해시가 manifest와 다르다")
            with np.load(io.BytesIO(data)) as d:
                sample = {key: d[key] for key in d.files}
            problem = sample_problem(sample)
            if problem:
                skipped.append({"asset_id": asset, "patient_pid": entry["patient_pid"], "split": entry["split"],
                                "reason": problem})
                continue
            name = scan_name(entry, sample_jaw(sample))
            save_sample(sample, cache / f"{name}.npz")
            no_normals += "normals" not in sample
        names[entry["split"]].append(name)
        scans[name] = entry
        if k % 100 == 0 or k == len(entries):
            print(f"[{k}/{len(entries)}] {(time.time() - t0) / 60:.1f} min  skipped {len(skipped)}", flush=True)

    (root / "splits").mkdir(exist_ok=True)
    for split, members in names.items():
        (root / "splits" / f"{split}.txt").write_text("".join(f"{n}\n" for n in members), encoding="utf-8")
    for extra in ("dataset.toml", "README.md"):
        (root / extra).write_bytes(_get(cli, f"{dataset}/{version}/{extra}"))
    (root / "manifest.jsonl").write_bytes(manifest)
    meta = {
        "dataset": dataset, "version": version,
        "definition": tomllib.loads((root / "dataset.toml").read_text(encoding="utf-8")),
        "scans": scans, "skipped": skipped,
    }
    (root / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    counts = ", ".join(f"{split} {len(members)}" for split, members in names.items())
    print(f"{dataset}/{version}: {counts}, skipped {len(skipped)} -> {root}")
    for item in skipped[:20]:
        print("  skipped", item["asset_id"], item["reason"])
    if no_normals:
        print(f'  주의: {no_normals}개 스캔에 normal이 없다.  ToothSegConfig.features에서 "normal", "curvature"를 뺀다')


def _load(root: Path, out: DownloadDataOutput) -> None:
    """다 받은 폴더 -> out."""
    splits = {split: (root / "splits" / f"{split}.txt").read_text(encoding="utf-8").split() for split in SPLITS}
    out.root, out.cache = root, root / "cache"
    out.train, out.val, out.test = splits["train"], splits["val"], splits["test"]
    out.meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
