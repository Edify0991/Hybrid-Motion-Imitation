"""Read-only kinematic coverage audit; not a policy performance evaluation."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
motion_dir = ROOT / 'src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/movebox'
records = []
for path in sorted(motion_dir.glob('*.npz')):
    with np.load(path, allow_pickle=False) as d:
        q = d['joint_pos'][:, 3:7]  # wxyz, per MotionLoader schema
        yaw = np.arctan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]), 1-2*(q[:,2]**2+q[:,3]**2))
        vel = d['object_lin_vel_w'][:, :2]
        angle = np.arctan2(vel[:,1], vel[:,0])-yaw
        angle = np.arctan2(np.sin(angle), np.cos(angle))
        moving = np.linalg.norm(vel, axis=1)>0.1
        edges = np.linspace(-180., 180., 9)
        hist = np.histogram(np.rad2deg(angle[moving]), bins=edges)[0]
        fps = float(d['fps'].item())
        records.append(dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            frames=len(q),fps=fps,duration_s=(len(q)-1)/fps,
            object_displacement_m=(d['object_pos_w'][-1]-d['object_pos_w'][0]).tolist(),
            object_z_start_end_max_m=[float(d['object_pos_w'][0,2]),float(d['object_pos_w'][-1,2]),float(d['object_pos_w'][:,2].max())],
            moving_frames=int(moving.sum()),direction_edges_deg=edges.tolist(),direction_counts=hist.tolist(),
            caveat='All frames, speed > 0.1 m/s; object velocity relative to root yaw. No verified hold/contact segmentation; not policy capability.'))
output = Path(__file__).with_name('motion_coverage.json')
output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(records, ensure_ascii=False, indent=2))
