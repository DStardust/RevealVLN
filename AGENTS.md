# Workspace instructions

- `FROZEN_SPEC.md` is the canonical Method-Freeze-2 research specification. Do not change its frozen claims, modules, datasets, protocols, or submission gates without an explicitly versioned novelty, feasibility, or correctness revision.
- A large placeholder file created by the user in this directory reserves `/mnt` capacity. Treat it as user-owned data: do not delete, truncate, move, compress, or reuse it unless the user explicitly asks to release that reserved space.
- The project must be self-contained under `/mnt/daiyang/vla`. Do not depend on, copy from, symlink to, or run against data, environments, checkpoints, code, or caches in other project/user directories. Acquire every external asset independently from its authoritative source and record its provenance inside this project.
