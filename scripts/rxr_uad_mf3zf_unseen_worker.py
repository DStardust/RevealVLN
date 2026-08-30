#!/usr/bin/env python3
"""MF3ZF specialization of the audited single-episode unseen worker."""

from __future__ import annotations

import rxr_uad_mf3zb_unseen_worker as worker


worker.REVISION = "mf3zf"
# The uncertainty comparator remains MF3V's frozen native-margin control.
worker.GATE = worker.base.MF3V_GATE


if __name__ == "__main__":
    raise SystemExit(worker.main())
