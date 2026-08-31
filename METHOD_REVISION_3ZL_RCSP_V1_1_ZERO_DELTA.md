# MF3ZL-RCSP v1.1 — zero-relative-delta correctness revision

This is a versioned correctness revision of `mf3zl_rcsp_v1`.  The sealed v1
implementation rejected an event whenever the frozen native and runner-up
embedding difference had zero norm.  Six of the 997 already sealed RxR
development events have that legitimate representation (`0.6%`).

The revision does not change the proposal hierarchy, labels, utility, loss,
risk constraint, folds, optimizer, or decision boundary.  It changes only the
representation contract:

* finite non-zero instruction, history, native, and runner embeddings remain
  required;
* a zero `runner - native` vector is retained as an explicit “no relative
  semantic evidence” state;
* its normalized relative vector is the zero vector, so the policy-side causal
  scalars remain usable while the relative semantic compatibility term is
  neutral;
* no row is dropped, no future observation is introduced, and no threshold is
  tuned from outcomes.

The original `revealnav_mf3/rcsp.py`, its v1 protocol, and its failed diagnostic
are immutable.  This revision is only authorized for the sealed RxR-only
development diagnostic until the full expanded-data protocol is separately
sealed.  It cannot authorize confirmation or any public split.

## Interpretation

This rule is a representation-domain correction, not an algorithmic rescue:
the model is explicitly told that the relative semantic channel contains no
directional evidence and must rely on the remaining causal policy features.
The six identities are recorded in the v1.1 probe protocol before fitting.
