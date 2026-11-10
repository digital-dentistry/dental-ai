# dental-ai

* 치아 3D Segmentation 모델 개발을 목표로 한다.

## How to use?

저장소 루트에서 `pip install -e . --no-deps`로 설치하고 `dental_ai <명령>`으로 실행한다.
컨테이너(`docker compose exec hdl-app bash`)에는 이미 설치돼 있다 — 아래 예시는 저장소 루트(`/app`) 기준 경로다.

```bash
dental_ai --help                     # 명령 목록
dental_ai <명령> --help              # 명령별 인자

# 데이터셋 원본 -> data/ios/<데이터셋>/raw (label 내장 PLY) (dental_ai.dataset.ToothSegRawImport)
# 한 번만 한다.  지금의 raw는 이미 만들어 두었다 — 원본은 data/ios/<데이터셋>/_zips에서 다시 푼다
dental_ai import-raw 3d_iosseg --src <풀어 둔 3D-IOSSeg> --out data/ios/3d_iosseg

# raw PLY -> npz 캐시 (dental_ai.dataset.ToothSegPreProcessing)
dental_ai preprocess data/ios/teeth3ds data/ios/3d_iosseg data/ios/poseidon3d data/ios/cbct_ios_fusion_liu2023

# 전처리된 캐시를 스캔마다 그림으로 -> runs/preprocess_view/<스캔>.png (dental_ai.dataset.ToothSegPreProcessView)
dental_ai preprocess-view --cache data/ios/3d_iosseg/cache --name IOSSEG007_upper IOSSEG007_lower
dental_ai preprocess-view --cache data/ios/teeth3ds/cache --count 0      # 캐시 전부

# DentalDataHub 데이터셋 릴리스 (S3) -> data/ios/ddh_<데이터셋>_<버전>/cache, splits (dental_ai.dataset.DDHLoader)
# pip install boto3 (컨테이너에는 있다).  환경 변수: DDH_S3_ENDPOINT=http://<서버>:9000, DDH_TRAINER_ACCESS_KEY, DDH_TRAINER_SECRET_KEY
dental_ai ddh-fetch ios-teethseg-full v1

# 받은 스캔을 브라우저에서 돌려 보기 -> runs/ddh_view/<데이터셋>_<버전>/<스캔>.html (dental_ai.visualizer.ddh_viewer)
# 보기: 3D / Color / Instance / FDI.  HTML 한 파일이라 화면 없는 서버에서 만들고 다른 PC에서 연다
dental_ai ddh-view ios-teethseg-full v1 --split val --count 4
```

데이터 구조는 `data/ios/README.md`에 있다.