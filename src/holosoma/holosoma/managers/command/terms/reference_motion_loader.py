"""Strict, opt-in single-pool motion loading for reference preview commands."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


REQUIRED = {"fps", "joint_names", "body_names", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"}


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest_pool(manifest_path, pool, robot_body_names, robot_joint_names, dt, device="cpu"):
    # Import here to avoid a module cycle with wbt.MotionCommand.
    from holosoma.managers.command.terms.wbt import MotionLoader, MultiMotionLoader

    path = Path(manifest_path).resolve(strict=True)
    manifest = json.loads(path.read_text())
    if manifest.get("version") not in {"reference-motion-v1", "abc-motion-v1"}:
        raise ValueError("unknown reference motion manifest version")
    fps = float(manifest["target_fps"])
    if not np.isfinite(fps) or fps <= 0 or not np.isfinite(float(dt)) or float(dt) <= 0:
        raise ValueError("invalid fps or control dt")
    if not np.isclose(fps * float(dt), 1.0, rtol=0, atol=1e-6):
        raise ValueError(f"manifest fps {fps} does not match control dt {dt}")
    if manifest.get("object_reference") is not False:
        raise ValueError("reference preview requires robot-only manifest")
    urdf = Path(manifest["target_urdf"]).resolve(strict=True)
    if sha(urdf) != manifest["target_urdf_sha256"]:
        raise ValueError("target URDF hash mismatch")
    joints = manifest["joint_names"]
    bodies = manifest["body_names"]
    if len(joints) != len(set(joints)) or len(bodies) != len(set(bodies)):
        raise ValueError("duplicate source schema names")
    if not set(robot_joint_names).issubset(joints) or not set(robot_body_names).issubset(bodies):
        raise ValueError("target simulator schema missing from manifest")
    entries = manifest["pools"].get(pool)
    if not entries:
        raise ValueError(f"empty or unknown pool: {pool}")
    loaders = []
    for entry in entries:
        if entry.get("object_reference") is not False:
            raise ValueError("object-reference clip in robot-only pool")
        clip = Path(entry["path"]).resolve(strict=True)
        if entry.get("sidecar"):
            sidecar = Path(entry["sidecar"]).resolve(strict=True)
            if sha(sidecar) != entry["sidecar_sha256"]:
                raise ValueError(f"sidecar hash mismatch: {sidecar}")
        if sha(clip) != entry["sha256"]:
            raise ValueError(f"clip hash mismatch: {clip}")
        with np.load(clip, allow_pickle=False) as data:
            if not REQUIRED.issubset(data.files):
                raise ValueError(f"missing NPZ fields: {clip}")
            if any(key.startswith("object_") for key in data.files):
                raise ValueError(f"unexpected object fields: {clip}")
            if not np.isclose(float(data["fps"]), fps, atol=1e-6):
                raise ValueError(f"clip fps mismatch: {clip}")
            if data["joint_names"].tolist() != joints or data["body_names"].tolist() != bodies:
                raise ValueError(f"clip name schema mismatch: {clip}")
            n = int(entry["frames"])
            if n < 2:
                raise ValueError(f"too few frames: {clip}")
            expected = {"joint_pos": (n, len(joints) + 7), "joint_vel": (n, len(joints) + 6), "body_pos_w": (n, len(bodies), 3), "body_quat_w": (n, len(bodies), 4), "body_lin_vel_w": (n, len(bodies), 3), "body_ang_vel_w": (n, len(bodies), 3)}
            for key, shape in expected.items():
                if data[key].shape != shape or not np.isfinite(data[key]).all():
                    raise ValueError(f"invalid {key} in {clip}")
            if np.any(np.abs(np.linalg.norm(data["joint_pos"][:, 3:7], axis=-1) - 1) > 1e-3) or np.any(np.abs(np.linalg.norm(data["body_quat_w"], axis=-1) - 1) > 1e-3):
                raise ValueError(f"unnormalized quaternion: {clip}")
        loaders.append(MotionLoader(str(clip), robot_body_names, robot_joint_names, device=device))
    result = MultiMotionLoader.__new__(MultiMotionLoader)
    lengths = torch.tensor([x.time_step_total for x in loaders], dtype=torch.long, device=device)
    result._motion_end_idx = lengths.cumsum(0)
    result._motion_start_idx = result._motion_end_idx - lengths
    result._num_motions = len(loaders)
    result._joint_pos = torch.cat([x.joint_pos for x in loaders], dim=0)
    result._joint_vel = torch.cat([x.joint_vel for x in loaders], dim=0)
    for key in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
        setattr(result, "_" + key, torch.cat([getattr(x, key) for x in loaders], dim=0))
    result._joint_indexes = torch.arange(len(robot_joint_names), dtype=torch.long, device=device)
    result._body_indexes = torch.arange(len(robot_body_names), dtype=torch.long, device=device)
    result._object_pos_w = torch.zeros(0, 3, device=device)
    result._object_quat_w = torch.zeros(0, 4, device=device)
    result._object_lin_vel_w = torch.zeros(0, 3, device=device)
    result._object_ang_vel_w = torch.zeros(0, 3, device=device)
    result.has_object = False
    result.fps = fps
    result.time_step_total = int(lengths.sum())
    return result
