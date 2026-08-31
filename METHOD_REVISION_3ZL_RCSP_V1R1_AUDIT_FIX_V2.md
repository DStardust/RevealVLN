# MF3ZL-RCSP v1r1 audit correction v2

The first audit-correction attempt exposed a second schema/provenance issue:
R2R `episode_id` values are only unique within a scene.  The same numeric ID
may legitimately occur in different MP3D scenes, so an episode-ID-only overlap
test is invalid.  The v2 audit uses the complete event identity
`(dataset, scene_id, episode_id, decision_step)` for conflict and overlap
checks.  It also relies on the sealed manifest-level public-split flag rather
than requiring a field that is not present on each record.

This remains a read-only, versioned audit correction.  No rollout, label,
selection, threshold, or data-gate rule is changed; the earlier failed audit
artifacts are retained.
