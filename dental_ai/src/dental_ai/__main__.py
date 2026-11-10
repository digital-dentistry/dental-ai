"""`dental_ai <명령> [인자]` — dental_ai 모듈의 실행 스크립트를 한 명령으로 모은 진입점.

    dental_ai import-raw 3d_iosseg --src data/open/3d_iosseg --out data/ios/3d_iosseg
    dental_ai preprocess data/ios/teeth3ds data/ios/3d_iosseg
    dental_ai preprocess-view --cache data/ios/3d_iosseg/cache --name IOSSEG007_upper
    dental_ai ddh-fetch ios-teethseg-full v1
    dental_ai ddh-view ios-teethseg-full v1 --split val

    dental_ai --help             # 명령 목록
    dental_ai <명령> --help       # 그 모듈의 인자

명령은 모듈을 따로 된 python 프로세스 (`python -m <모듈>`)로 그대로 실행한다: 인자는 손대지
않고 넘기고, 상대 경로는 지금 디렉터리 기준이다.
"""
from __future__ import annotations

import subprocess
import sys

# 명령 -> (실행할 모듈, 설명)
COMMANDS: dict[str, tuple[str, str]] = {
    "import-raw": ("dental_ai.dataset.ToothSegRawImport", "IOS 데이터셋 원본 -> data/ios/<데이터셋>/raw (label 내장 PLY)"),
    "preprocess": ("dental_ai.dataset.ToothSegPreProcessing", "data/ios/<데이터셋>/raw PLY -> cache npz"),
    "preprocess-view": ("dental_ai.dataset.ToothSegPreProcessView", "npz 캐시 (전처리 결과)를 스캔마다 그림으로"),
    "ddh-fetch": ("dental_ai.dataset.DDHLoader", "DDH 데이터셋 릴리스 (S3) -> data/ios/ddh_<데이터셋>_<버전> cache npz"),
    "ddh-view": ("dental_ai.visualizer.ddh_viewer", "DDH에서 받은 스캔 -> 브라우저 3D 뷰어 HTML (3D / Color / Instance / FDI)"),
}


def usage() -> str:
    width = max(map(len, COMMANDS))
    lines = [f"  {name:<{width}}  {desc}" for name, (_, desc) in COMMANDS.items()]
    return "usage: dental_ai <명령> [인자]   (명령마다 --help)\n\n명령:\n" + "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(usage())
        return 0
    name, rest = argv[0], argv[1:]
    if name not in COMMANDS:
        print(f"dental_ai: 모르는 명령 {name!r}\n\n{usage()}", file=sys.stderr)
        return 2
    try:
        return subprocess.call([sys.executable, "-m", COMMANDS[name][0], *rest])
    except KeyboardInterrupt:                    # Ctrl-C는 자식도 받는다: 자식이 끝나면 조용히 나간다
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
