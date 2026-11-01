# Development workflow

`origin` is the personal fork; `upstream` is the original repository. Work on `dev`, review `git diff`, stage explicit source/config/test/document paths, commit, then `git push origin dev`. Pull uses fast-forward only.

## Local dependencies

The 579 exact paths in `dependency_assets.txt` are intentionally excluded from the dev index. They include seven NPZ motions, five climb joint-position NPY motions, six ONNX inference policies, 142 OBJ meshes, 310 STL meshes, 49 DAE meshes, 22 MTL materials, 24 scene/robot XML models and 14 URDF models. Python package markers and the small `largebox.urdf.jinja` template remain tracked as code contracts. README demonstration media remain tracked.

After a fresh clone of dev:

```sh
git remote add upstream https://github.com/jiashunwang/Hybrid-Motion-Imitation.git
git fetch upstream
python scripts/restore_dependency_assets.py
git ls-files -ci --exclude-standard
```

The helper reads upstream blobs and exclusively creates missing paths. Existing files, including local modifications, are untouched, and nothing is staged. It does not access external original data directories. Use the upstream revision compatible with this branch (initial baseline c353731) when reproducibility is needed.

## Synchronization

Avoid switching this asset-populated dev worktree between branches: ignored dependencies may collide with tracked upstream files. Use a separate clean worktree:

```sh
git fetch upstream
git worktree add ../upstream-sync main
git -C ../upstream-sync merge --ff-only upstream/main
# Optional: publish the unchanged upstream main to the personal fork
git -C ../upstream-sync push origin main
```

To integrate upstream into dev, first review incoming changes in the clean worktree, then merge upstream/main on a dedicated integration branch/worktree. Upstream changes may reintroduce assets or cause modify/delete conflicts; resolve each asset intentionally, retain local copies, and remove unwanted paths from the index only. Recheck `git ls-files -ci --exclude-standard`, the asset manifest and `git diff --cached` before committing. Never force-push shared dev history.
