#!/usr/bin/env python3
"""Train-only worker for exact MF3ZF expanded-band return collection."""

from __future__ import annotations

import os


os.environ["REVEALNAV_MF3ZF_COLLECTION_ONLY"] = "1"

import rxr_uad_controller_worker_mf3 as worker


if __name__ == "__main__":
    worker.main()
