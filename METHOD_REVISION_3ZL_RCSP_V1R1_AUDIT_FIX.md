# MF3ZL-RCSP v1r1 audit correction

The sealed v1r1 collection and all 290 targeted treatments completed
successfully.  Its original audit helper stopped on a schema mismatch while
reading the older canonical rows: those rows store `identity.decision_step`,
not `identity.step`.

This revision is a read-only audit correction.  It does not rerun rollouts,
change labels, alter the collection population, or relax any data gate.  It
uses the sealed v1r1 protocol and manifests, resolves the documented field
name, and counts identities after conflict checking.  It writes a separate
audit artifact; the original collector and its sealed protocol remain
immutable.
