#!/usr/bin/env python3
"""MF3ZE specialization of the audited single-episode unseen worker."""

from __future__ import annotations

import rxr_uad_mf3zb_unseen_worker as worker


worker.REVISION = "mf3ze"
# The secondary treatment remains MF3V's frozen native-margin control; the
# MF3ZE train-only gate applies only to the ensemble treatment.
worker.GATE = worker.base.MF3V_GATE


if __name__ == "__main__":
    raise SystemExit(worker.main())
