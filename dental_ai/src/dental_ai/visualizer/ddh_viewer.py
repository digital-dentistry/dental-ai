"""DDH에서 받은 스캔을 브라우저에서 돌려 보는 3D 뷰어.  스캔마다 HTML 파일 하나를 쓴다.

    dental_ai ddh-view ios-teethseg-full v1                        # 앞의 8개 -> runs/ddh_view/<릴리스>/<스캔>.html
    dental_ai ddh-view ios-teethseg-full v1 --split val --count 0   # val 전부
    dental_ai ddh-view ios-teethseg-full v1 --name <스캔 이름> ...
    (= python -m dental_ai.visualizer.ddh_viewer <인자>)

릴리스는 ddh_fetch로 연다: 이미 받았으면 S3에 묻지 않고, 아직이면 먼저 받는다 (dataset/DDHLoader.py).
HTML 한 파일에 데이터와 그리는 코드 (WebGL)가 다 들어 있어 인터넷이 필요 없다.  화면 없는 학습 서버에서
만들고 다른 PC의 브라우저로 연다.  스캔 하나가 2~3 MB다.

보기 (버튼 또는 1~4 키).  모든 보기에 음영이 들어간다:

    3D        한 가지 색.  normal이 없는 릴리스 (ios-teethseg v1)는 이웃 점으로 추정한 normal로 음영을 준다
    Color     스캔 색 (rgb).  색이 없는 스캔은 버튼이 꺼진다
    Instance  치아 instance (ins)마다 다른 색, 치은은 회색
    FDI       cls의 FDI 번호.  색은 preprocess-view와 같다 (악궁 위치마다 한 색, 이웃한 치아는 서로 다른 색)

Instance, FDI는 오른쪽 범례에 마우스를 올리면 그 치아만 밝게 남는다.
마우스: 왼쪽 끌기 회전, 오른쪽 (또는 Shift) 끌기 이동, 휠 확대 (커서 쪽으로).  처음 보기는 교합면 (전치가
위)이고, O / F 키나 버튼으로 교합면 / 정면 보기로 돌아간다.

노트북이나 학습 코드에서는 write_view로 캐시 하나를 바로 쓴다:

    from dental_ai.visualizer.ddh_viewer import write_view
    name = ddh.val[0]
    write_view(ddh.cache / f"{name}.npz", Path("runs/ddh_view/a.html"), ddh.meta["scans"][name])
"""
from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from ..dataset.algorithm.download_data.download_data import DownloadDataOutput
from ..dataset.ToothSegPreProcessView import CLASS_RGB, GINGIVA, SLOTS, CachedScan, class_to_fdi, load_cache, view_frames

PLAIN = "#e8e4da"                                    # 3D 보기의 색.  DDH 대시보드 뷰어와 같다
ENTRY_KEYS = ("asset_id", "patient_pid", "split", "label_version")   # manifest 한 줄에서 보여 주는 값


def _hex(h: str) -> list[int]:
    return [int(h[i:i + 2], 16) for i in (1, 3, 5)]


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def point_sizes(q: np.ndarray, tree: cKDTree) -> np.ndarray:
    """[V] 점 지름 (0.01 mm 단위, uint8).  셋째 이웃까지 거리의 1.5배라, 스캐너가 성기게 찍은 치은·구개에도
    틈이 보이지 않고 촘촘한 치아는 작은 점으로 남는다.  확대하면 점도 함께 커진다."""
    k = min(4, len(q))
    distance = tree.query(q, k=k)[0][:, -1] if k > 1 else np.zeros(len(q))
    return np.clip(np.round(distance * 1.5 * 100), 5, 255).astype(np.uint8)


def pca_normals(q: np.ndarray, tree: cKDTree, k: int = 12) -> np.ndarray:
    """[V,3] normal 없는 스캔의 음영용 normal: 이웃 k점 local PCA의 가장 얇은 축.  부호는 정하지 않는다
    (뷰어는 |n·z|만 쓴다).  학습 feature로는 쓰지 않는다."""
    k = min(k, len(q))
    if k < 3:
        return np.zeros_like(q)
    near = q[tree.query(q, k=k)[1]]                                   # [V,k,3]
    d = near - near.mean(1, keepdims=True)
    return np.linalg.eigh(np.einsum("nki,nkj->nij", d, d))[1][:, :, 0]   # 가장 작은 고유값의 고유벡터


def _legend(scan: CachedScan) -> dict[str, list[dict]]:
    """보기마다 범례 줄 {key, label, count}.  key는 cls (FDI) / ins (Instance) 값이다."""
    def label(c: int) -> str:
        fdi = class_to_fdi(c, scan.jaw_upper)
        return {"gingiva": "치은", "?": "? (위치 모름)"}.get(fdi, fdi)

    fdi = [{"key": int(c), "label": label(int(c)), "count": int((scan.classes == c).sum())}
           for c in np.unique(scan.classes)]
    instance = []
    for i in np.unique(scan.instances):
        member = scan.classes[scan.instances == i]
        name = "치은" if i == 0 else f"#{i}  {label(int(np.bincount(member).argmax()))}"
        instance.append({"key": int(i), "label": name, "count": int(len(member))})
    return {"fdi": fdi, "instance": instance}


def has_normals(scan: CachedScan) -> bool:
    """캐시 nrm이 전부 0이면 normal 없는 릴리스다 (ios-teethseg v1)."""
    return scan.zero_normals < len(scan.xyz)


def scan_data(scan: CachedScan, info: list[tuple[str, str]]) -> dict:
    """HTML에 넣는 값.  좌표는 교합면 보기의 화면 축으로 돌려 둔다 (처음 보기 = 회전 없음)."""
    occlusal, front = view_frames(scan)
    q = (scan.xyz - scan.xyz.mean(0)) @ occlusal.T
    tree = cKDTree(q)
    normals = scan.normals @ occlusal.T if has_normals(scan) else pca_normals(q, tree)
    return {
        "title": scan.name,
        "info": [[k, v] for k, v in info],
        "n": len(q),
        "radius": float(np.linalg.norm(q, axis=1).max()),
        "pos": _b64(q.astype("<f4")),
        "size": _b64(point_sizes(q, tree)),
        "nrm": _b64(np.round(normals * 127).astype(np.int8)),
        "rgb": _b64(np.round(scan.rgb * 255).astype(np.uint8)) if scan.rgb is not None else None,
        "cls": _b64(scan.classes.astype(np.uint8)),
        "ins": _b64(scan.instances.astype(np.uint8)),
        "front": (front @ occlusal.T).flatten().tolist(),     # 교합면 화면 축 -> 정면 화면 축 (행 우선)
        "plain": _hex(PLAIN),
        "gingiva": _hex(GINGIVA),
        "classColours": np.round(CLASS_RGB * 255).astype(int).tolist(),
        "instanceColours": [_hex(h) for h in SLOTS],
        "legend": _legend(scan),
    }


def scan_html(scan: CachedScan, info: list[tuple[str, str]]) -> str:
    """캐시 한 스캔 -> HTML 문서 하나."""
    data = json.dumps(scan_data(scan, info), ensure_ascii=False).replace("</", "<\\/")   # </script>로 끊기지 않게
    return PAGE.replace("__TITLE__", html.escape(scan.name)).replace("__DATA__", data)


def write_view(npz: Path, out: Path, entry: dict | None = None, release: str = "") -> Path:
    """캐시 npz 하나 -> out (HTML).  entry는 manifest 한 줄 (DownloadDataOutput.meta["scans"][이름])."""
    scan = load_cache(Path(npz))
    info = [("릴리스", release)] if release else []
    info += [(k, str(entry[k])) for k in ENTRY_KEYS if entry and k in entry]
    info += [("턱", "위턱" if scan.jaw_upper else "아래턱"), ("점", f"{len(scan.xyz):,}"),
             ("normal", "있음" if has_normals(scan) else "없음 (음영은 이웃 점으로 추정)"),
             ("색", "있음" if scan.rgb is not None else "없음")]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(scan_html(scan, info), encoding="utf-8")
    return out


def write_views(release: DownloadDataOutput, names: list[str], out_dir: Path) -> list[Path]:
    """받은 릴리스의 스캔들 -> out_dir/<이름>.html."""
    tag = f"{release.meta['dataset']}/{release.meta['version']}"
    return [write_view(release.cache / f"{name}.npz", Path(out_dir) / f"{name}.html", release.meta["scans"].get(name), tag)
            for name in names]


def pick(release: DownloadDataOutput, split: str, names: list[str] | None, count: int) -> list[str]:
    """그릴 스캔 이름.  names를 주면 그것만 (split, count는 보지 않는다)."""
    every = release.train + release.val + release.test
    if names:
        unknown = [n for n in names if n not in every]
        if unknown:
            raise SystemExit(f"릴리스에 없는 스캔: {', '.join(unknown)}  (이름은 {release.root}/splits/*.txt)")
        return names
    pool = every if split == "all" else getattr(release, split)
    return pool if count <= 0 else pool[:count]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help="DDH 데이터셋 이름 (예: ios-teethseg-full)")
    ap.add_argument("version", help="릴리스 버전 (예: v1)")
    ap.add_argument("--data", type=Path, default=Path("data/ios"), help="ddh_fetch의 out: 받은 릴리스가 있는 곳")
    ap.add_argument("--split", choices=("all", "train", "val", "test"), default="all")
    ap.add_argument("--name", nargs="+", default=None, help="스캔 이름.  주면 --split, --count는 보지 않는다")
    ap.add_argument("--count", type=int, default=8, help="앞에서 몇 개.  0이면 전부")
    ap.add_argument("--out", type=Path, default=Path("runs/ddh_view"), help="이 아래 <데이터셋>_<버전>/<스캔>.html")
    args = ap.parse_args()

    from ..dataset.DDHLoader import DDHError, ddh_fetch     # boto3가 필요하다.  write_view만 쓸 때는 없어도 된다
    try:
        release = ddh_fetch(args.dataset, args.version, args.data)
    except DDHError as e:
        raise SystemExit(str(e)) from e
    names = pick(release, args.split, args.name, args.count)
    out_dir = args.out / f"{args.dataset}_{args.version}"
    for path in write_views(release, names, out_dir):
        print(path)
    print(f"{len(names)}개 -> {out_dir}  (브라우저로 연다)")


# ------------------------------------------------------------------- 페이지
# 데이터는 <script type="application/json">에 넣는다.  좌표 float32, normal int8 (x127), 점 지름 uint8 (0.01 mm),
# 색·cls·ins uint8, 모두 base64.
PAGE = r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  [hidden] { display: none !important; }        /* 아래 display 규칙이 hidden을 이기지 않게 */
  html, body { margin: 0; height: 100%; overflow: hidden; background: #1c1c1b; color: #ecebe6;
               font: 13px/1.45 system-ui, -apple-system, "Segoe UI", "Malgun Gothic", sans-serif; }
  canvas { display: block; width: 100vw; height: 100vh; cursor: grab; touch-action: none; }
  canvas.dragging { cursor: grabbing; }
  .panel { position: absolute; background: rgba(28, 28, 27, 0.88); border: 1px solid #3a3a38;
           border-radius: 8px; padding: 10px 12px; }
  #side { top: 12px; left: 12px; max-width: 380px; }
  #legend { top: 12px; right: 12px; max-height: calc(100vh - 50px); overflow-y: auto; min-width: 160px; }
  h1 { font-size: 14px; margin: 0 0 6px; word-break: break-all; }
  table { border-collapse: collapse; color: #b9b8b0; }
  td { padding: 0 10px 0 0; vertical-align: top; }
  td + td { color: #ecebe6; word-break: break-all; }
  .row { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 10px; align-items: center; }
  button { font: inherit; color: #ecebe6; background: #2c2c2a; border: 1px solid #4a4a47; border-radius: 6px;
           padding: 3px 10px; cursor: pointer; }
  button[aria-pressed="true"] { background: #2a78d6; border-color: #2a78d6; color: #fff; }
  button:disabled { opacity: 0.35; cursor: default; }
  label { display: flex; gap: 6px; align-items: center; color: #b9b8b0; }
  .item { display: flex; align-items: center; gap: 8px; padding: 1px 4px; border-radius: 4px; }
  .item:hover { background: #3a3a38; }
  .swatch { width: 12px; height: 12px; border-radius: 3px; flex: none; }
  .count { margin-left: auto; padding-left: 14px; color: #898781; font-variant-numeric: tabular-nums; }
  .hint { margin-top: 10px; color: #898781; }
  #error { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; }
</style>
</head>
<body>
<canvas id="view"></canvas>
<div id="side" class="panel">
  <h1 id="title"></h1>
  <table id="info"></table>
  <div class="row" id="modes"></div>
  <div class="row">
    <button id="occlusal" title="O">교합면</button>
    <button id="front" title="F">정면</button>
    <label>점 크기 <input id="size" type="range" min="0.5" max="3" step="0.1" value="1"></label>
  </div>
  <div class="hint">왼쪽 끌기 회전 · 오른쪽/Shift 끌기 이동 · 휠 확대 · 1~4 보기 · O/F 처음 보기</div>
</div>
<div id="legend" class="panel" hidden></div>
<div id="error" hidden></div>
<script type="application/json" id="data">__DATA__</script>
<script>
"use strict";
const VS = `
attribute vec3 aPos;
attribute vec3 aNrm;
attribute vec3 aCol;
attribute float aSize;                         // 점 지름 (0.01 mm)
uniform mat3 uRot;
uniform vec2 uPan;
uniform vec2 uScale;
uniform float uRadius;
uniform float uPxPerMm;
uniform float uSize;                           // 점 크기 슬라이더 (배율)
varying vec3 vCol;
void main() {
  vec3 p = uRot * aPos;                        // 화면 축: x 오른쪽, y 위, z 카메라 쪽 (mm)
  gl_Position = vec4((p.xy + uPan) * uScale, -p.z / uRadius, 1.0);
  gl_PointSize = max(1.5, aSize * 0.01 * uPxPerMm * uSize);
  vCol = aCol * (0.4 + 0.6 * abs((uRot * aNrm).z));   // 카메라 쪽 조명.  뒤집힌 normal도 밝기는 같다
}`;
const FS = `
precision mediump float;
varying vec3 vCol;
void main() {
  vec2 d = gl_PointCoord - vec2(0.5);
  if (dot(d, d) > 0.25) discard;               // 둥근 점
  gl_FragColor = vec4(vCol, 1.0);
}`;

function fail(message) {
  const box = document.getElementById("error");
  box.textContent = message;
  box.hidden = false;
}

try {
  const D = JSON.parse(document.getElementById("data").textContent);
  const bytes = (b64) => {
    const s = atob(b64), a = new Uint8Array(s.length);
    for (let i = 0; i < s.length; i++) a[i] = s.charCodeAt(i);
    return a;
  };
  const N = D.n;
  const pos = new Float32Array(bytes(D.pos).buffer);
  const nrm = new Int8Array(bytes(D.nrm).buffer);
  const rgb = D.rgb ? bytes(D.rgb) : null;
  const cls = bytes(D.cls), ins = bytes(D.ins);

  // ------------------------------------------------------------ 패널
  document.getElementById("title").textContent = D.title;
  const info = document.getElementById("info");
  for (const [k, v] of D.info) {
    const tr = info.insertRow();
    tr.insertCell().textContent = k;
    tr.insertCell().textContent = v;
  }
  const MODES = [["3d", "3D"], ["color", "Color"], ["instance", "Instance"], ["fdi", "FDI"]];
  const buttons = {};
  for (const [key, label] of MODES) {
    const b = document.createElement("button");
    b.textContent = label;
    b.disabled = key === "color" && !rgb;
    b.title = b.disabled ? "이 스캔에는 색이 없다" : String(MODES.findIndex((m) => m[0] === key) + 1);
    b.onclick = () => setMode(key);
    document.getElementById("modes").appendChild(b);
    buttons[key] = b;
  }

  // ------------------------------------------------------------ 색
  let mode = "3d", highlight = null;
  const instanceColour = (k) => (k ? D.instanceColours[(k - 1) % D.instanceColours.length] : D.gingiva);
  const classColour = (k) => D.classColours[k] || D.classColours[D.classColours.length - 1];
  function colours() {
    const out = new Uint8Array(N * 3);
    for (let i = 0; i < N; i++) {
      let c = D.plain;
      if (mode === "color") c = [rgb[3 * i], rgb[3 * i + 1], rgb[3 * i + 2]];
      else if (mode === "fdi") c = classColour(cls[i]);
      else if (mode === "instance") c = instanceColour(ins[i]);
      const key = mode === "fdi" ? cls[i] : ins[i];
      const dim = highlight !== null && key !== highlight ? 0.2 : 1;
      out[3 * i] = c[0] * dim;
      out[3 * i + 1] = c[1] * dim;
      out[3 * i + 2] = c[2] * dim;
    }
    return out;
  }
  const legend = document.getElementById("legend");
  function showLegend() {
    const rows = D.legend[mode];
    legend.replaceChildren();
    legend.hidden = !rows;
    for (const row of rows || []) {
      const item = document.createElement("div");
      item.className = "item";
      const swatch = document.createElement("span");
      swatch.className = "swatch";
      const c = mode === "fdi" ? classColour(row.key) : instanceColour(row.key);
      swatch.style.background = `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
      const name = document.createElement("span");
      name.textContent = row.label;
      const count = document.createElement("span");
      count.className = "count";
      count.textContent = row.count.toLocaleString();
      item.append(swatch, name, count);
      item.onmouseenter = () => { highlight = row.key; upload(); };
      item.onmouseleave = () => { highlight = null; upload(); };
      legend.appendChild(item);
    }
  }

  // ------------------------------------------------------------ WebGL
  const canvas = document.getElementById("view");
  const gl = canvas.getContext("webgl", { antialias: true });
  if (!gl) throw new Error("이 브라우저는 WebGL을 쓸 수 없다");
  const compile = (type, src) => {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
  };
  const prog = gl.createProgram();
  gl.attachShader(prog, compile(gl.VERTEX_SHADER, VS));
  gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, FS));
  gl.bindAttribLocation(prog, 0, "aPos");
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
  gl.useProgram(prog);
  gl.enable(gl.DEPTH_TEST);
  const u = {};
  for (const name of ["uRot", "uPan", "uScale", "uRadius", "uPxPerMm", "uSize"]) u[name] = gl.getUniformLocation(prog, name);
  function attribute(name, data, components, type, normalized) {
    const buffer = gl.createBuffer(), at = gl.getAttribLocation(prog, name);
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(at);
    gl.vertexAttribPointer(at, components, type, normalized, 0, 0);
    return buffer;
  }
  attribute("aPos", pos, 3, gl.FLOAT, false);
  attribute("aSize", bytes(D.size), 1, gl.UNSIGNED_BYTE, false);
  attribute("aNrm", nrm, 3, gl.BYTE, true);
  const colourBuffer = attribute("aCol", new Uint8Array(N * 3), 3, gl.UNSIGNED_BYTE, true);

  // ------------------------------------------------------------ 보기
  const OCCLUSAL = [1, 0, 0, 0, 1, 0, 0, 0, 1];
  let R = OCCLUSAL.slice(), pan = [0, 0], zoom = 1, size = 1;
  const mul = (a, b) => {
    const o = [];
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++) o.push(a[3 * i] * b[j] + a[3 * i + 1] * b[3 + j] + a[3 * i + 2] * b[6 + j]);
    return o;
  };
  const rotX = (t) => [1, 0, 0, 0, Math.cos(t), -Math.sin(t), 0, Math.sin(t), Math.cos(t)];
  const rotY = (t) => [Math.cos(t), 0, Math.sin(t), 0, 1, 0, -Math.sin(t), 0, Math.cos(t)];
  const transpose = (m) => [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]];
  function orthonormal(m) {                       // 회전을 거듭해도 찌그러지지 않게 (Gram-Schmidt, 행)
    const x = m.slice(0, 3), y = m.slice(3, 6);
    const norm = (v) => { const l = Math.hypot(...v); return v.map((c) => c / l); };
    const nx = norm(x), d = nx[0] * y[0] + nx[1] * y[1] + nx[2] * y[2];
    const ny = norm(y.map((c, i) => c - d * nx[i]));
    const nz = [nx[1] * ny[2] - nx[2] * ny[1], nx[2] * ny[0] - nx[0] * ny[2], nx[0] * ny[1] - nx[1] * ny[0]];
    return [...nx, ...ny, ...nz];
  }
  const mmPerPixel = () => (2 * D.radius * 1.05) / zoom / Math.min(canvas.clientWidth, canvas.clientHeight);

  function draw() {
    const dpr = window.devicePixelRatio || 1;
    const w = Math.round(canvas.clientWidth * dpr), h = Math.round(canvas.clientHeight * dpr);
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    gl.viewport(0, 0, w, h);
    gl.clearColor(0.11, 0.11, 0.106, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    const half = (D.radius * 1.05) / zoom, m = Math.min(w, h);
    gl.uniformMatrix3fv(u.uRot, false, transpose(R));     // GLSL mat3은 열 우선
    gl.uniform2f(u.uPan, pan[0], pan[1]);
    gl.uniform2f(u.uScale, m / w / half, m / h / half);
    gl.uniform1f(u.uRadius, D.radius * 1.01);
    gl.uniform1f(u.uPxPerMm, m / (2 * half));
    gl.uniform1f(u.uSize, size);
    gl.drawArrays(gl.POINTS, 0, N);
  }
  let pending = false;
  function redraw() {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => { pending = false; draw(); });
  }
  function upload() {
    gl.bindBuffer(gl.ARRAY_BUFFER, colourBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, colours(), gl.DYNAMIC_DRAW);
    redraw();
  }
  function setMode(key) {
    if (buttons[key].disabled) return;
    mode = key;
    highlight = null;
    for (const k in buttons) buttons[k].setAttribute("aria-pressed", String(k === key));
    showLegend();
    upload();
  }
  function setView(rotation) {
    R = rotation.slice();
    pan = [0, 0];
    zoom = 1;
    redraw();
  }

  // ------------------------------------------------------------ 입력
  let drag = null;
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());
  canvas.addEventListener("pointerdown", (e) => {
    drag = { x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey };
    canvas.setPointerCapture(e.pointerId);
    canvas.classList.add("dragging");
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    drag.x = e.clientX;
    drag.y = e.clientY;
    if (drag.pan) {
      const k = mmPerPixel();
      pan = [pan[0] + dx * k, pan[1] - dy * k];
    } else {
      R = orthonormal(mul(rotY(dx * 0.01), mul(rotX(dy * 0.01), R)));
    }
    redraw();
  });
  const release = () => { drag = null; canvas.classList.remove("dragging"); };
  canvas.addEventListener("pointerup", release);
  canvas.addEventListener("pointercancel", release);
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const ux = e.clientX - rect.left - rect.width / 2, uy = rect.height / 2 - (e.clientY - rect.top);
    const before = mmPerPixel();
    zoom = Math.min(50, Math.max(0.2, zoom * Math.exp(-e.deltaY * 0.0015)));
    const after = mmPerPixel();
    pan = [pan[0] + ux * (after - before), pan[1] + uy * (after - before)];   // 커서 아래 점이 제자리에
    redraw();
  }, { passive: false });
  window.addEventListener("keydown", (e) => {
    if (e.target instanceof HTMLInputElement) return;
    const key = { 1: "3d", 2: "color", 3: "instance", 4: "fdi" }[e.key];
    if (key) setMode(key);
    else if (e.key === "o" || e.key === "O") setView(OCCLUSAL);
    else if (e.key === "f" || e.key === "F") setView(D.front);
  });
  window.addEventListener("resize", redraw);
  document.getElementById("occlusal").onclick = () => setView(OCCLUSAL);
  document.getElementById("front").onclick = () => setView(D.front);
  document.getElementById("size").oninput = (e) => { size = Number(e.target.value); redraw(); };

  setMode("3d");
} catch (e) {
  fail(String(e && e.message ? e.message : e));
}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
