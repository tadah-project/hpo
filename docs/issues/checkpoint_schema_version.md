# CORE Checkpoint META schema_version is never enforced

`CORE` `Checkpoint` writes a META section with `schema_version` but `CheckpointManager` / loaders do not
check it. HPO no longer writes checkpoints (the BASINS strategies and the `checkpoint.tadah` resume were
removed in 1.5.1), so nothing depends on it today. Any future user of the library needs a generic policy
first (refuse / migrate per section version, in one place) rather than per-section contract tags.
