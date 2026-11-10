from __future__ import annotations

import numpy as np

from .scan_encoder import ScanEncoderInput, ScanEncoderOutput

INT32_MAX = 2 ** 31 - 1


def ply_encoder_cpu(inp: ScanEncoderInput, out: ScanEncoderOutput) -> bool:
    """binary little-endian PLY 한 파일 (data/ios raw 형식).

        element vertex: float x y z, [float nx ny nz], [uchar red green blue], [int fdi, int label, int instance]
        element face:   list uchar int vertex_indices

    label은 정점 속성이다 — face로 label을 준 데이터셋도 import할 때 정점 다수결로 바꾼다.
    fdi / label / instance는 DentalLabeler가 쓰는 필드와 같은 이름 · 값이다.
    임시 파일에 쓰고 이름을 바꾸므로 중간에 멈춰도 반쯤 쓴 PLY가 남지 않는다.
    False: 정점이 없거나, 길이가 안 맞거나, label 셋 중 일부만 있거나, int32를 넘는다.
    """
    n = inp.positions.shape[0]
    label_parts = (inp.fdi, inp.labels, inp.instances)
    labelled = inp.fdi is not None
    if n == 0 or any((t is not None) != labelled for t in label_parts):
        return False
    for t in (inp.normals, inp.colours, *label_parts):
        if t is not None and t.shape[0] != n:
            return False
    if labelled and any(int(t.min()) < -INT32_MAX or int(t.max()) > INT32_MAX for t in label_parts):
        return False

    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if inp.normals is not None:
        fields += [("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")]
    if inp.colours is not None:
        fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    if labelled:
        fields += [("fdi", "<i4"), ("label", "<i4"), ("instance", "<i4")]
    vert = np.empty(n, np.dtype(fields))
    xyz = inp.positions.numpy().astype(np.float32)
    vert["x"], vert["y"], vert["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if inp.normals is not None:
        nrm = inp.normals.numpy().astype(np.float32)
        vert["nx"], vert["ny"], vert["nz"] = nrm[:, 0], nrm[:, 1], nrm[:, 2]
    if inp.colours is not None:
        rgb = np.round(inp.colours.clamp(0, 1).numpy() * 255).astype(np.uint8)
        vert["red"], vert["green"], vert["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    if labelled:
        vert["fdi"], vert["label"], vert["instance"] = (t.numpy() for t in label_parts)

    faces = np.empty(0, np.dtype([("n", "u1"), ("i", "<i4", 3)]))
    if inp.faces is not None and inp.faces.numel():
        faces = np.empty(inp.faces.shape[0], faces.dtype)
        faces["n"], faces["i"] = 3, inp.faces.numpy().astype(np.int32)

    names = {"<f4": "float", "u1": "uchar", "<i4": "int"}
    header = ["ply", "format binary_little_endian 1.0"]
    header += [f"comment {c}".replace("\n", " ") for c in inp.comments]
    header += [f"element vertex {n}"] + [f"property {names[t]} {k}" for k, t in fields]
    header += [f"element face {len(faces)}", "property list uchar int vertex_indices", "end_header"]
    data = ("\n".join(header) + "\n").encode("ascii") + vert.tobytes() + faces.tobytes()

    inp.path.parent.mkdir(parents=True, exist_ok=True)
    tmp = inp.path.with_name(inp.path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(inp.path)
    out.size = len(data)
    return True
