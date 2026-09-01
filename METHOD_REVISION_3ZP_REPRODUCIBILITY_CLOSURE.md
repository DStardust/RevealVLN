# MF3ZP Reproducibility Closure

This versioned correctness closure preserves the historical MF3ZP protocols
byte-for-byte while allowing reviewed descendants of base commit
`3e16465d095e4e8ae36ad4ea310f6e02fc9737b1`.

Verification requires the base commit to be an ancestor of `HEAD`, exact hashes
for the original formal protocol, v1.1 correctness protocol, frozen 300-event
records and selection, the 538-record status, aggregate inventories for frozen
instruction and evidence records, and every scout implementation file. It also
requires all public flags and downstream authorizations to remain closed.

The closure never rewrites the historical protocol's `source_commit`; it checks
scientific files by content rather than requiring `HEAD` to equal an obsolete
parent commit. Existing results are immutable and no result file is overwritten.
