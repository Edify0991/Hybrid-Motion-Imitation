import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from convert_reference_motions import URDFFK, clock, resample, velocities
from holosoma.managers.command.terms.reference_motion_loader import load_manifest_pool, sha
from holosoma.managers.command.terms.wbt import MotionCommand, MotionLoader
from holosoma.config_values.wbt.g1.command import motion_config, reference_preview_motion_config
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "generated_data/reference_motions_50hz/manifest.json"


def test_clock_no_tail_padding():
    t = clock(4, 30, 50)
    assert np.allclose(t, [0, .02, .04, .06, .08, .10])
    assert t[-1] <= 3 / 30


def test_resample_translation_sign_flip_and_world_angular_velocity():
    src_t = np.arange(61) / 60
    p = np.column_stack((2 * src_t, np.zeros(61), np.ones(61)))
    angle = src_t * 0.4
    q = np.column_stack((np.cos(angle / 2), np.zeros(61), np.zeros(61), np.sin(angle / 2)))
    q[::2] *= -1
    out_p, out_q, out_j, t = resample(p, q, src_t[:, None], 60, 50)
    linear, angular = velocities(out_p, out_q, 50)
    assert np.max(np.abs(linear[:, 0] - 2)) < 1e-10
    assert np.max(np.abs(angular[:, 2] - .4)) < 1e-10
    assert np.allclose(out_j[:, 0], t)
    assert np.allclose(np.linalg.norm(out_q, axis=1), 1)


def test_shortest_arc_across_pi():
    q = np.array([[np.cos(np.deg2rad(179)/2), 0, 0, np.sin(np.deg2rad(179)/2)],
                  [np.cos(np.deg2rad(181)/2), 0, 0, np.sin(np.deg2rad(181)/2)]])
    _, out_q, _, _ = resample(np.zeros((2, 3)), q, np.zeros((2, 1)), 1, 2)
    _, w = velocities(np.zeros((len(out_q), 3)), out_q, 2)
    assert np.allclose(w[:, 2], np.deg2rad(2), atol=1e-9)


def test_world_angular_velocity_from_tilted_start():
    t = np.arange(11) / 50
    rot = Rotation.from_euler("z", .5 * t) * Rotation.from_euler("x", .7)
    q = rot.as_quat()[:, [3, 0, 1, 2]]
    _, omega = velocities(np.zeros((11, 3)), q, 50)
    assert np.allclose(omega, np.tile([0, 0, .5], (11, 1)), atol=1e-10)


def test_fk_fixed_hand_and_real_pool():
    m = json.loads(MANIFEST.read_text())
    model = URDFFK(m["target_urdf"])
    with np.load(m["pools"]["amass_locomotion"][0]["path"], allow_pickle=False) as d:
        p, q, j = d["joint_pos"][:, :3], d["joint_pos"][:, 3:7], d["joint_pos"][:, 7:]
        names, body, _, _, _ = model.compute(p[:3], q[:3], j[:3], 50)
        assert "left_sphere_hand_link" in names
        assert np.allclose(body, d["body_pos_w"][:3], atol=1e-5)


def test_manifest_loader_permuted_names_and_failfast(tmp_path):
    m = json.loads(MANIFEST.read_text())
    joints, bodies = m["joint_names"][::-1], m["body_names"][::-1] + ["left_ankle_roll_link"]
    x = load_manifest_pool(MANIFEST, "amass_locomotion", bodies, joints, .02)
    assert x.num_motions == 13 and x.time_step_total == sum(e["frames"] for e in m["pools"]["amass_locomotion"])
    assert x.body_pos_w.shape[1] == len(bodies)
    with np.load(m["pools"]["amass_locomotion"][0]["path"], allow_pickle=False) as d:
        assert np.allclose(x.joint_pos[0].numpy(), d["joint_pos"][0, 7:][::-1])
    with pytest.raises(ValueError, match="fps"):
        load_manifest_pool(MANIFEST, "amass_locomotion", bodies, joints, .01)
    for mutation, message in [
        (lambda y: y["pools"]["amass_locomotion"][0].update(sha256="0"*64), "hash"),
        (lambda y: y["pools"]["amass_locomotion"][0].update(object_reference=True), "object"),
        (lambda y: y["pools"]["amass_locomotion"][0].update(frames=1), "frames"),
    ]:
        altered = json.loads(MANIFEST.read_text())
        mutation(altered)
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(altered))
        with pytest.raises(ValueError, match=message):
            load_manifest_pool(path, "amass_locomotion", bodies, joints, .02)


def test_sidecar_contact_event_is_provenance_only():
    m = json.loads(MANIFEST.read_text())
    entries = m["pools"]["physhsi_interaction_aux"]
    assert len(entries) == 8
    assert all(e["object_reference"] is False for e in entries)
    pickup = next(e for e in entries if "/pickUp/" in e["source"])
    side = json.loads(Path(pickup["sidecar"]).read_text())
    event = side["contact_event"]
    assert event["output_frame"] == round(event["source_frame"] * 50 / 60)
    assert "not measured" in side["source_semantics"]


def test_legacy_and_opt_in_command_hook():
    assert motion_config.enable_terminal_pose_hold is True
    assert reference_preview_motion_config.enable_terminal_pose_hold is False
    legacy = ROOT / "src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj.npz"
    with np.load(legacy, allow_pickle=False) as d:
        bodies, joints = d["body_names"].tolist(), d["joint_names"].tolist()
    original = MotionLoader(str(legacy), bodies, joints)
    assert original.time_step_total > 2
    m = json.loads(MANIFEST.read_text())
    cmd = object.__new__(MotionCommand)
    cmd.motion_cfg = reference_preview_motion_config
    cmd._env = SimpleNamespace(dt=.02)
    cmd.device = "cpu"
    selected = cmd._load_motion_data(m["body_names"], m["joint_names"])
    assert selected.num_motions == 13 and selected.time_step_total == 3699


def test_corrupt_sidecar_fails_fast(tmp_path):
    m = json.loads(MANIFEST.read_text())
    entry = m["pools"]["physhsi_interaction_aux"][0].copy()
    m["pools"] = {"single": [entry]}
    side = tmp_path / "wrong.json"
    side.write_text("{}")
    entry["sidecar"] = str(side)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="sidecar hash"):
        load_manifest_pool(path, "single", m["body_names"], m["joint_names"], .02)


@pytest.mark.parametrize("damage,match", [("missing", "missing NPZ"), ("nan", "invalid joint_pos"), ("fps", "clip fps")])
def test_corrupt_clip_fails_fast(tmp_path, damage, match):
    m = json.loads(MANIFEST.read_text())
    entry = m["pools"]["amass_locomotion"][0].copy()
    m["pools"] = {"single": [entry]}
    with np.load(entry["path"], allow_pickle=False) as source:
        arrays = {k: source[k].copy() for k in source.files}
    if damage == "missing":
        del arrays["body_ang_vel_w"]
    elif damage == "nan":
        arrays["joint_pos"][0, 0] = np.nan
    else:
        arrays["fps"] = np.array(49.)
    clip = tmp_path / "bad.npz"
    np.savez(clip, **arrays)
    entry.update(path=str(clip), sha256=sha(clip))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(m))
    with pytest.raises(ValueError, match=match):
        load_manifest_pool(manifest, "single", m["body_names"], m["joint_names"], .02)
