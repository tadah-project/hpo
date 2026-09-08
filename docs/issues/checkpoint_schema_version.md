# checkpoint.tadah META schema_version is never enforced

`CORE` `Checkpoint` writes a META section with `schema_version` but `CheckpointManager` / loaders do not
check it. Phase 1 added per-section contract tags for the basin pools (`params_real`,
`space_fingerprint`) with explicit refusal on mismatch; a generic policy (refuse / migrate per section
version, one place) would cover HPO_Host, HPOLog and future sections uniformly.
