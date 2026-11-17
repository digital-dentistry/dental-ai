"""data/ios/<데이터셋>/raw의 label 내장 PLY -> ToothSegLoader가 읽는 npz 캐시 (cache/).

    dental_ai preprocess data/ios/teeth3ds data/ios/3d_iosseg      # 데이터셋 폴더 여러 개
    (= python -m dental_ai.dataset.ToothSegPreProcessing <인자>)

raw PLY는 dental_ai import-raw가 데이터셋 원본에서 만든다 (ToothSegRawImport.py).
결과 확인: dental_ai preprocess-view --cache data/ios/3d_iosseg/cache  (ToothSegPreProcessView.py)

스캔 하나 = npz 하나.  원본 정점을 전부 남긴다 (채점은 full-resolution ground truth가
필요하다).  downsample은 loader가 하므로 학습 해상도를 바꿔도 다시 돌릴 필요가 없다.
label이 없는 raw PLY (3DTeethLand 테스트 스캔 등)는 건너뛴다.

    xyz [V,3] float32 mm, nrm [V,3] float16 (면적 가중 mesh normal), cls [V] uint8 (0 gingiva,
    1..16, 17 위치 모르는 치아), ins [V] uint8 (0 gingiva, 1..K), jaw uint8 (1 upper),
    rgb [V,3] uint8 — 실제 색이 있을 때만

색: 실제 정점 색이 있으면 rgb로 저장한다.  Teeth3DS의 0.502 회색처럼 한 가지 색뿐이면
색이 없는 것으로 본다.  색 없는 스캔은 loader가 매 epoch 치아/치은별로 칠한다
(algorithm/augmentation/synth_colour_cpu.py, in-house 실측 컬러맵).
"""
from __future__ import annotations

import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from .algorithm.augmentation.augmentation import UNKNOWN_TOOTH
from .algorithm.normal_estimation.mesh_normals_cpu import mesh_normals_cpu
from .algorithm.normal_estimation.normal_estimation import NormalEstimationInput, NormalEstimationOutput
from .algorithm.scan_decoder.ply_decoder_cpu import ply_decoder_cpu
from .algorithm.scan_decoder.scan_decoder import ScanDecoderInput, ScanDecoderOutput, VertexLabel


# ------------------------------------------------------------------ labels
def fdi_to_class(fdi: np.ndarray, jaw_upper: bool) -> np.ndarray:
    """FDI 번호 -> 0 gingiva, 1..8 (위턱 1사분면 / 아래턱 3사분면), 9..16 (2 / 4사분면),
    UNKNOWN_TOOTH (그 턱의 번호가 아니거나 위치가 1..8이 아닌 치아).  ToothSegLoader의 class다."""
    fdi = fdi.astype(np.int64)
    quadrant, tooth = fdi // 10, fdi % 10
    first, second = (1, 2) if jaw_upper else (3, 4)
    valid = (tooth >= 1) & (tooth <= 8)
    out = np.full_like(fdi, UNKNOWN_TOOTH)
    out[valid & (quadrant == first)] = tooth[valid & (quadrant == first)]
    out[valid & (quadrant == second)] = tooth[valid & (quadrant == second)] + 8
    out[fdi == 0] = 0
    return out


def ordered_instances(instances: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """치아 instance id -> 1..K, 치아마다 다수결 class 순서.  같은 스캔은 언제나 같은 id가 된다."""
    teeth = [i for i in np.unique(instances) if i != 0]
    teeth.sort(key=lambda i: int(np.bincount(classes[instances == i]).argmax()))
    out = np.zeros(len(instances), np.uint8)
    for k, i in enumerate(teeth, start=1):
        out[instances == i] = k
    return out


def jaw_of(name: str) -> bool:
    """'<이름>_upper' / '<이름>_lower' -> 위턱인가."""
    if name.endswith("_upper"):
        return True
    if name.endswith("_lower"):
        return False
    raise ValueError(f"{name}: raw names end in _upper or _lower")


# ------------------------------------------------------------------ colour
def real_colour(colours: torch.Tensor | None) -> np.ndarray | None:
    """[V,3] 0..1 -> uint8 RGB.  색이 없거나 한 가지 색뿐이면 (Teeth3DS의 회색) None."""
    if colours is None or float(colours.std(0, correction=0).mean()) < 0.01:
        return None
    return np.round(colours.clamp(0.0, 1.0).numpy() * 255).astype(np.uint8)


# ------------------------------------------------------------------- cache
def cache_arrays(positions: torch.Tensor, normals: torch.Tensor, classes: np.ndarray, instances: np.ndarray,
                 jaw_upper: bool, rgb: np.ndarray | None) -> dict[str, np.ndarray]:
    """캐시 npz에 들어가는 배열 (dtype까지 같다).  디스크에 쓰지 않고 메모리에서 읽는 DDHStream도 이것을 쓴다."""
    arrays = dict(xyz=positions.numpy().astype(np.float32), nrm=normals.numpy().astype(np.float16),
                  cls=classes.astype(np.uint8), ins=instances.astype(np.uint8), jaw=np.uint8(1 if jaw_upper else 0))
    if rgb is not None:
        arrays["rgb"] = rgb
    return arrays


def save_cache(path: Path, positions: torch.Tensor, normals: torch.Tensor, classes: np.ndarray,
               instances: np.ndarray, jaw_upper: bool, rgb: np.ndarray | None) -> None:
    np.savez(path, **cache_arrays(positions, normals, classes, instances, jaw_upper, rgb))


def cache_scan(ply_path: Path, out_path: Path) -> tuple[str | None, str | None]:
    """raw PLY 하나 -> npz.  (오류, 참고) 메시지를 돌려준다."""
    try:
        jaw_upper = jaw_of(ply_path.stem)
        mesh = ScanDecoderOutput()
        if not ply_decoder_cpu(ScanDecoderInput(ply_path), mesh) or mesh.faces is None:
            return "could not decode a mesh", None
        if mesh.fdi is None:
            return None, "skipped: no labels"
        labels = mesh.labels.numpy()
        other = sorted(set(np.unique(labels).tolist()) - {VertexLabel.GINGIVA, VertexLabel.TOOTH})
        if other:                                # 치은으로 섞어 넣지 않는다 — cache class에 자리를 먼저 만든다
            return f"label {other} (-1 unlabeled, 2 scan body ...) has no cache class yet", None
        nrm = NormalEstimationOutput()
        if not mesh_normals_cpu(NormalEstimationInput(mesh.positions, mesh.faces), nrm):
            return "no faces to estimate normals from", None
        # 치아인지는 label이 정한다.  번호(fdi)가 없는 치아는 위치 모르는 치아 (UNKNOWN_TOOTH)다
        tooth = labels == VertexLabel.TOOTH
        cls = fdi_to_class(np.where(tooth, mesh.fdi.numpy(), 0), jaw_upper)
        cls[tooth & (cls == 0)] = UNKNOWN_TOOTH
        save_cache(out_path, mesh.positions, nrm.normals, cls, ordered_instances(mesh.instances.numpy(), cls),
                   jaw_upper, real_colour(mesh.colours))
        return None, None
    except Exception as e:                       # 깨진 스캔 하나가 전체를 멈추면 안 된다
        return repr(e), None


def _run(job: tuple[Path, Path]) -> tuple[Path, str | None, str | None]:
    return (job[1], *cache_scan(*job))


def _worker_init() -> None:
    torch.set_num_threads(1)                     # worker가 여럿이니 worker마다 1 thread


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", type=Path, nargs="+", help="data/ios/<데이터셋> 폴더 (raw/ -> cache/)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="데이터셋마다 앞의 N개 스캔만 (smoke test)")
    ap.add_argument("--force", action="store_true", help="이미 있는 캐시도 다시 만든다")
    a = ap.parse_args()

    jobs = []
    for d in a.datasets:
        # 스캔은 <이름>_<jaw>.ply뿐이다.  <이름>.labeled.ply 같은 다른 도구의 부속 파일 (DentalLabeler가
        # 원본 옆에 쓰는 작업 파일)은 건너뛴다
        plys = sorted(p for p in (d / "raw").glob("*.ply") if "." not in p.stem)
        if not plys:
            raise SystemExit(f"{d / 'raw'}: no .ply (import first: dental_ai import-raw)")
        (d / "cache").mkdir(exist_ok=True)
        jobs += [(p, d / "cache" / f"{p.stem}.npz") for p in plys[:a.limit or None]]
    todo = [j for j in jobs if a.force or not j[1].exists()]
    print(f"{len(jobs)} scans, {len(todo)} to process", flush=True)
    t0, errs, skipped = time.time(), [], 0
    with Pool(a.workers, initializer=_worker_init) as pool:
        for k, (dst, err, note) in enumerate(pool.imap_unordered(_run, todo, chunksize=4), 1):
            if err:
                errs.append((dst, err))
            skipped += note is not None
            if k % 100 == 0 or k == len(todo):
                el = time.time() - t0
                print(f"[{k}/{len(todo)}] {el / 60:.1f} min  eta {(len(todo) - k) * el / k / 60:.1f} min  "
                      f"errs {len(errs)}", flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min, {len(errs)} errors, {skipped} unlabelled scans skipped")
    for d, e in errs[:20]:
        print("  error", d.name, e)


if __name__ == "__main__":
    main()
