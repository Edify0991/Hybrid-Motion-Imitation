# Reference motion conversion and loader handoff (2026-11-01)

This implements the **data and loading portion** of the ABC-light plan. It does not implement the A interaction bank, ABC rewards, reset logic, contact handling, or policy training. The earlier broad plan's LAFAN-first step is superseded here by the selected 13 AMASS walking clips and 8 PhysHSI carry/pickup/putdown clips.

## Files

- `reference_motion_conversion.yaml`: explicit read-only roots, shortlist, pool selectors, target URDF, 50 Hz, height policy, and 0.01 rad source limit tolerance.
- `convert_reference_motions.py`: CPU conversion, shortest-arc quaternion interpolation, SO(3) world angular velocity, target URDF forward kinematics (including fixed sphere hands), output manifest and PhysHSI provenance sidecars.
- `src/holosoma/holosoma/managers/command/terms/reference_motion_loader.py`: strict single-pool loader. Checks manifest-referenced clip/sidecar/model hashes, schemas, finite arrays, FPS, control dt and object-reference validity; gathers each clip's joints and bodies before concatenation. The manifest itself has no external signature.
- `src/holosoma/holosoma/config_types/command.py` and `src/holosoma/holosoma/managers/command/terms/wbt.py`: optional `motion_manifest`/`motion_pool` integration. The legacy source path is unchanged when the manifest is empty. Manifest mode forbids automatic prepend, append, terminal hold, fixed scene source, and mixed legacy file/dir selection.
- `src/holosoma/holosoma/config_values/{command,experiment}.py` and `.../wbt/g1/{command,experiment}.py`: CLI-registered `g1_29dof_wbt_reference_preview` robot-only single-pool preset using the halfspherehand asset. This is for inspection, not ABC training.
- `test_reference_motion_pipeline.py`: CPU checks of time grid, SLERP and quaternion sign, angular velocity across pi, target FK, per-clip name normalization, validation failures, contact provenance.

## Reproduce

From `/home/user/wmd/GfR-ABC-Light-20261101`:

```bash
/home/user/anaconda3/envs/hssim/bin/python convert_reference_motions.py reference_motion_conversion.yaml
PYTHONPATH=src/holosoma /home/user/anaconda3/envs/hssim/bin/python -m pytest -q test_reference_motion_pipeline.py
PYTHONPATH=src/holosoma /home/user/anaconda3/envs/hssim/bin/python - <<'PY'
import json
from holosoma.managers.command.terms.reference_motion_loader import load_manifest_pool
m = json.load(open('generated_data/reference_motions_50hz/manifest.json'))
for pool in m['pools']:
    x = load_manifest_pool('generated_data/reference_motions_50hz/manifest.json', pool, m['body_names'], m['joint_names'], .02)
    print(pool, x.num_motions, x.time_step_total, x.has_object)
PY
```

The generated manifest records absolute source paths and SHA-256, source and target clock information, output and sidecar hashes, URDF/config/shortlist hashes, joint/body schema, endpoint positions, joint-limit exceedances, and pending physical validation. Output is robot-only. PhysHSI `contact_index` is a **single source event frame**; its sidecar maps it to a nearest output frame and does not claim a measured contact mask. The box sidecar preserves the source local position and height, while no object orientation or object reference is synthesized. The root height policy follows the original PhysHSI loader's `base_height + 0.05`; velocities are recomputed after this adjustment. AMASS root and joints retain their source positions except interpolation. The uniform target clock stops at or before the final source instant; there is no duplicate terminal frame.

Generated data are ignored by Git. AMASS pool: 13 clips, 3699 frames. PhysHSI interaction auxiliary pool: 8 clips, 1373 frames. Both use 29 joints and 42 unique target bodies. Eleven shortlisted `loco_direction` clips remain optional by adding a separate pool selector to the YAML. Existing source joint-limit excesses within 0.01 rad are preserved and recorded; larger excesses fail conversion. Neither CPU loading nor source audit proves contact dynamics, collision validity, or training quality.

To inspect a one-pool preview config, import `holosoma.config_values.experiment.DEFAULTS['g1_29dof_wbt_reference_preview']`. The Tyro CLI selector `exp:g1-29dof-wbt-reference-preview` parses successfully. To select the auxiliary pool, append `--command.setup-terms.motion-command.params.motion-config.motion-pool physhsi_interaction_aux`; this override also parses successfully while retaining the halfspherehand robot asset and `enable_terminal_pose_hold=False`. No simulator rollout was run in this handoff.

Independent regression verification ran `test_reference_motion_pipeline.py`, `src/holosoma/holosoma/config_types/tests/test_command.py`, `src/holosoma/holosoma/config_values/tests/test_wbt_hybrid_climb.py`, and `src/holosoma/holosoma/config_values/tests/test_wbt_climb.py`: **23 passed in 1.30 s**. The Tyro default preview and auxiliary-pool override were also parsed with `tyro.cli`.

To roll back, revert the six tracked source hooks and remove the converter, config, strict loader, test, and this handoff file; delete `generated_data/reference_motions_50hz` if the generated files are no longer needed. The original source repositories and `/data` were not modified.
