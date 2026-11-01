"""Convert selected AMASS and PhysHSI G1 clips to robot-only Holosoma motions.

Run from the repository root with the hssim Python environment. No simulator needed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation, Slerp

VERSION = "reference-motion-v1"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clock(n, src_fps, dst_fps):
    # Last sample is the latest target instant not after the last source sample.
    return np.arange(int(np.floor((n - 1) * dst_fps / src_fps + 1e-9)) + 1) / dst_fps


def normalize_wxyz(q):
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("zero quaternion")
    q = q / norm
    for i in range(1, len(q)):
        if np.dot(q[i - 1], q[i]) < 0:
            q[i] *= -1
    return q


def resample(root, quat_wxyz, joints, src_fps, dst_fps):
    t = np.arange(len(root)) / src_fps
    target = clock(len(root), src_fps, dst_fps)
    pos = np.column_stack([np.interp(target, t, root[:, j]) for j in range(3)])
    q = normalize_wxyz(quat_wxyz)
    rotations = Rotation.from_quat(q[:, [1, 2, 3, 0]])
    out_q = Slerp(t, rotations)(target).as_quat()[:, [3, 0, 1, 2]]
    out_q = normalize_wxyz(out_q)
    # Joint limits are enforced separately; never unwrap limited revolute joints.
    out_j = np.column_stack([np.interp(target, t, joints[:, j]) for j in range(joints.shape[1])])
    return pos, out_q, out_j, target


def velocities(pos, quat_wxyz, fps):
    linear = np.gradient(pos, 1 / fps, axis=0, edge_order=1)
    rot = Rotation.from_quat(quat_wxyz[:, [1, 2, 3, 0]])
    angular = np.empty((len(pos), 3))
    if len(pos) < 2:
        raise ValueError("motion requires at least two frames")
    angular[0] = (rot[1] * rot[0].inv()).as_rotvec() * fps
    angular[-1] = (rot[-1] * rot[-2].inv()).as_rotvec() * fps
    if len(pos) > 2:
        angular[1:-1] = (rot[2:] * rot[:-2].inv()).as_rotvec() * (fps / 2)
    return linear, angular


class URDFFK:
    def __init__(self, path):
        tree = ET.parse(path).getroot()
        self.links = [e.attrib["name"] for e in tree.findall("link")]
        self.joints = []
        self.limits = {}
        for e in tree.findall("joint"):
            name, kind = e.attrib["name"], e.attrib["type"]
            if kind not in ("fixed", "revolute", "continuous") or e.find("mimic") is not None:
                raise ValueError(f"unsupported target joint: {name} ({kind})")
            parent = e.find("parent").attrib["link"]
            child = e.find("child").attrib["link"]
            if parent not in ["pelvis", *(j[3] for j in self.joints)]:
                raise ValueError(f"non-topological URDF parent: {parent}")
            origin = e.find("origin")
            xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
            rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
            axis_elem = e.find("axis")
            axis = np.fromstring(axis_elem.attrib.get("xyz", "1 0 0"), sep=" ") if axis_elem is not None else np.array([1., 0., 0.])
            self.joints.append((name, kind, parent, child, xyz, Rotation.from_euler("xyz", rpy), axis))
            limit = e.find("limit")
            if kind == "revolute" and limit is not None:
                self.limits[name] = (float(limit.attrib["lower"]), float(limit.attrib["upper"]))
        self.dof_names = [j[0] for j in self.joints if j[1] != "fixed"]
        if len(self.dof_names) != 29 or self.links[0] != "pelvis":
            raise ValueError("unexpected target G1 URDF")

    def compute(self, root_pos, root_quat, q, fps):
        n = len(q)
        poses = {}
        root_r = Rotation.from_quat(root_quat[:, [1, 2, 3, 0]])
        poses["pelvis"] = (root_pos, root_r)
        columns = {name: i for i, name in enumerate(self.dof_names)}
        for name, kind, parent, child, xyz, origin_r, axis in self.joints:
            pp, pr = poses[parent]
            offset = np.broadcast_to(xyz, (n, 3))
            cp = pp + pr.apply(offset)
            cr = pr * origin_r
            if kind != "fixed":
                cr = cr * Rotation.from_rotvec(q[:, columns[name], None] * axis)
            poses[child] = (cp, cr)
        names = list(self.links)
        bp = np.stack([poses[name][0] for name in names], axis=1)
        bq = np.stack([poses[name][1].as_quat()[:, [3, 0, 1, 2]] for name in names], axis=1)
        bl = np.stack([velocities(bp[:, i], bq[:, i], fps)[0] for i in range(len(names))], axis=1)
        ba = np.stack([velocities(bp[:, i], bq[:, i], fps)[1] for i in range(len(names))], axis=1)
        return names, bp, bq, bl, ba


def source_clip(path, kind, mapping, height_policy):
    if kind == "amass":
        with np.load(path, allow_pickle=False) as d:
            names = d["dof_names"].tolist()
            body = d["body_names"].tolist()
            pelvis = body.index("pelvis")
            return d["body_positions"][:, pelvis].astype(float), d["body_rotations"][:, pelvis].astype(float), d["dof_positions"].astype(float), names, float(np.asarray(d["fps"]).item()), None
    d = torch.load(path, map_location="cpu", weights_only=True)
    names = list(mapping)
    q = np.asarray(d["joint_position"].cpu(), dtype=float)
    if q.shape[1] != len(names):
        raise ValueError("PhysHSI mapping width mismatch")
    p = np.asarray(d["base_position"].cpu(), dtype=float).copy()
    h = np.asarray(d["base_height"].cpu(), dtype=float)
    if height_policy == "physhsi_loader":
        p[:, 2] = h + 0.05
    elif height_policy == "base_height":
        p[:, 2] = h
    elif height_policy != "base_position":
        raise ValueError("unknown height policy")
    quat = np.asarray(d["base_quat"].cpu(), dtype=float)[:, [3, 0, 1, 2]]
    sidecar = {}
    for key in ("base_position", "base_height", "box_pos_local", "box_height_global", "contact_index"):
        if key in d:
            value = d[key]
            sidecar[key] = np.asarray(value.cpu() if hasattr(value, "cpu") else value).tolist()
    return p, quat, q, names, 60.0, sidecar


def main(config_path):
    cfg_path = Path(config_path).resolve()
    config = yaml.safe_load(cfg_path.read_text())
    if config["source_quaternion"] != {"amass": "WXYZ", "physhsi": "XYZW"}:
        raise ValueError("source quaternion contract mismatch: AMASS WXYZ, PhysHSI XYZW")
    if not np.isfinite(float(config["target_fps"])) or float(config["target_fps"]) <= 0:
        raise ValueError("target_fps must be finite and positive")
    repo = Path(__file__).resolve().parent
    shortlist_path = Path(config["selection_manifest"])
    if not shortlist_path.is_absolute():
        shortlist_path = repo / shortlist_path
    shortlist = json.loads(shortlist_path.read_text())
    out = (repo / config["output_dir"]).resolve()
    existing_manifest = out / "manifest.json"
    if existing_manifest.exists() and json.loads(existing_manifest.read_text()).get("config_sha256") != sha(cfg_path):
        raise FileExistsError(f"configuration differs from existing manifest; choose a new output_dir: {out}")
    urdf_path = repo / config["target_urdf"]
    model = URDFFK(urdf_path)
    mapping_file = Path(config["physhsi_mapping"])
    mapping = {}
    for line in mapping_file.read_text().splitlines():
        index, name = line.split()
        mapping[name] = int(index)
    if set(mapping) != set(model.dof_names):
        raise ValueError("PhysHSI mapping does not match target URDF")
    source_roots = {"amass": Path(config["amass_root"]), "physhsi": Path(config["physhsi_root"])}
    selected = []
    for spec in config["pools"]:
        kind = spec["source"]
        if kind not in source_roots:
            raise ValueError(f"unknown source: {kind}")
        entries = shortlist["candidates"] if kind == "amass" else shortlist["physhsi"]
        selected += [(kind, spec["name"], e["path"]) for e in entries if any(e["path"].startswith(prefix) for prefix in spec["path_prefixes"])]
    if not selected:
        raise ValueError("selection produced no clips")
    pools = {name: [] for _, name, _ in selected}
    fps = float(config["target_fps"])
    for kind, pool, relative in selected:
        source = (source_roots[kind] / relative).resolve()
        p, quat, q, names, source_fps, sidecar = source_clip(source, kind, mapping, config["physhsi_height_policy"])
        if not np.isclose(source_fps, float(config["source_fps"][kind]), atol=1e-6):
            raise ValueError(f"unexpected source fps: {source}")
        if len(set(names)) != len(names) or set(names) != set(model.dof_names):
            raise ValueError(f"joint names mismatch: {source}")
        q = q[:, [names.index(name) if kind == "amass" else mapping[name] for name in model.dof_names]]
        original_n = len(q)
        p, quat, q, times = resample(p, quat, q, source_fps, fps)
        limit_violations = {}
        for j, name in enumerate(model.dof_names):
            if name in model.limits:
                lo, hi = model.limits[name]
                excess = max(float(lo - q[:, j].min()), float(q[:, j].max() - hi), 0.0)
                if excess > 0:
                    limit_violations[name] = dict(min=float(q[:, j].min()), max=float(q[:, j].max()), lower=lo, upper=hi)
                if excess > float(config["joint_limit_tolerance_rad"]):
                    raise ValueError(f"joint limit excess {excess:.5f} > tolerance for {name}: {source}")
        root_v, root_w = velocities(p, quat, fps)
        joint_v = np.gradient(q, 1 / fps, axis=0, edge_order=1)
        body_names, bp, bq, bv, bw = model.compute(p, quat, q, fps)
        arrays = dict(fps=np.array(fps), joint_names=np.array(model.dof_names), body_names=np.array(body_names), joint_pos=np.column_stack((p, quat, q)).astype("f4"), joint_vel=np.column_stack((root_v, root_w, joint_v)).astype("f4"), body_pos_w=bp.astype("f4"), body_quat_w=bq.astype("f4"), body_lin_vel_w=bv.astype("f4"), body_ang_vel_w=bw.astype("f4"))
        if any(not np.isfinite(v).all() for v in arrays.values() if np.issubdtype(v.dtype, np.number)):
            raise ValueError(f"nonfinite data: {source}")
        dest = out / pool / Path(relative).with_suffix(".npz")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            with np.load(dest, allow_pickle=False) as old:
                if set(old.files) != set(arrays) or any(not np.array_equal(old[k], v) for k, v in arrays.items()):
                    raise FileExistsError(f"existing output differs: {dest}")
        else:
            np.savez_compressed(dest, **arrays)
        sidecar_path = None
        if sidecar is not None:
            sidecar_path = dest.with_suffix(".source.json")
            if "contact_index" in sidecar:
                event = int(sidecar["contact_index"])
                sidecar["contact_event"] = dict(source_frame=event, source_time_s=event / source_fps, output_frame=int(round(event * fps / source_fps)), rounding="nearest", valid=0 <= event < original_n)
            sidecar.update(source_semantics="PhysHSI source metadata only; box orientation and physical contact are not measured", height_policy=config["physhsi_height_policy"])
            encoded = json.dumps(sidecar)
            if sidecar_path.exists() and sidecar_path.read_text() != encoded:
                raise FileExistsError(f"existing sidecar differs: {sidecar_path}")
            sidecar_path.write_text(encoded)
        pools[pool].append(dict(path=str(dest), sha256=sha(dest), source=str(source), source_sha256=sha(source), source_frames=original_n, source_fps=source_fps, frames=len(times), last_time_s=float(times[-1]), sidecar=str(sidecar_path) if sidecar_path else None, sidecar_sha256=sha(sidecar_path) if sidecar_path else None, joint_limit_violations=limit_violations, object_reference=False, physical_validation="pending", source_quaternion=config["source_quaternion"][kind], output_quaternion="WXYZ", output_joint_schema="URDF actuated joint order", source_range=[0, original_n-1], first_root_position=p[0].tolist(), last_root_position=p[-1].tolist()))
        print(pool, relative, original_n, "->", len(times), flush=True)
    manifest = dict(version=VERSION, target_fps=fps, target_urdf=str(urdf_path), target_urdf_sha256=sha(urdf_path), config_sha256=sha(cfg_path), shortlist_sha256=sha(shortlist_path), joint_names=model.dof_names, body_names=model.links, pools=pools, object_reference=False, physical_validation="pending")
    manifest_path = out / "manifest.json"
    encoded = json.dumps(manifest, indent=2)
    if manifest_path.exists():
        if manifest_path.read_text() != encoded:
            raise FileExistsError(f"existing manifest differs; choose a new output_dir: {manifest_path}")
    else:
        temporary = out / "manifest.json.tmp"
        temporary.write_text(encoded)
        temporary.replace(manifest_path)
    print("manifest", out / "manifest.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    main(parser.parse_args().config)
