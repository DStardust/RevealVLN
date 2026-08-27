#!/usr/bin/env python3
"""Validator for Reveal Prefix hash-chain traces.

Recomputes the whole chain from the fixed genesis and verifies every link.
`compare` checks two chains (e.g. cold run vs fresh-process replay) for
prefix count, candidate cardinality, candidate identity/mapping, action
trace and chain-root equality.

Exit codes: 0 = PASS, 1 = FAIL, 2 = usage/IO error.
"""

import argparse
import hashlib
import json
import sys

GENESIS_HASH = hashlib.sha256(
    b"RevealNav-Phase0-Reveal-Prefix-Genesis-v1").hexdigest()
SCHEMA_VERSION = "reveal-prefix-trace/1"


def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def load_chain(path):
    with open(path) as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def verify_chain(chain):
    """Recompute the chain from scratch; return (ok, root, problems)."""
    problems = []
    prev = GENESIS_HASH
    root = None
    for k, rec in enumerate(chain):
        if rec.get("schema_version") != SCHEMA_VERSION:
            problems.append("record %d: bad schema_version" % k)
        if rec.get("previous_record_hash") != prev:
            problems.append("record %d: previous_record_hash mismatch "
                            "(expected %s...)" % (k, prev[:12]))
        stored = rec.get("current_record_hash")
        recomputed_rec = {kk: vv for kk, vv in rec.items()
                          if kk != "current_record_hash"}
        recomputed = sha256_bytes(
            canonical_json(recomputed_rec).encode("utf-8"))
        if stored != recomputed:
            problems.append("record %d: current_record_hash mismatch "
                            "(stored %s... recomputed %s...)"
                            % (k, str(stored)[:12], recomputed[:12]))
        # internal sub-hash consistency
        cand_identity = rec.get("candidates")
        if cand_identity is not None:
            expect = sha256_bytes(canonical_json(cand_identity).encode("utf-8"))
            if rec.get("candidate_set_hash") != expect:
                problems.append("record %d: candidate_set_hash mismatch" % k)
        if rec.get("observation_hashes") is not None:
            expect = sha256_bytes(canonical_json(
                rec["observation_hashes"]).encode("utf-8"))
            if rec.get("observation_hash") != expect:
                problems.append("record %d: observation_hash mismatch" % k)
        if rec.get("action") is not None:
            expect = sha256_bytes(canonical_json(
                rec["action"]).encode("utf-8"))
            if rec.get("action_hash") != expect:
                problems.append("record %d: action_hash mismatch" % k)
        graph = rec.get("graph") or {}
        if graph:
            expect = sha256_bytes(canonical_json({
                "cur_vp": graph.get("cur_vp"),
                "mappings": graph.get("mappings"),
                "loc_noise": graph.get("loc_noise"),
            }).encode("utf-8"))
            if rec.get("graph_mapping_hash") != expect:
                problems.append("record %d: graph_mapping_hash mismatch" % k)
            if "loc_noise" not in graph:
                problems.append("record %d: graph.loc_noise missing" % k)
        prev = stored
        root = stored
    ok = not problems and len(chain) > 0
    return ok, root, problems


def compare_chains(chain_a, chain_b):
    checks = []

    def check(name, ok, observed):
        checks.append({"name": name, "pass": bool(ok),
                       "observed": str(observed)[:300]})

    check("prefix count equal", len(chain_a) == len(chain_b),
          (len(chain_a), len(chain_b)))
    n = min(len(chain_a), len(chain_b))
    cardinality_ok = True
    identity_ok = True
    mapping_ok = True
    action_ok = True
    first_div = None
    for k in range(n):
        ra, rb = chain_a[k], chain_b[k]
        ca, cb = ra.get("candidates"), rb.get("candidates")
        if (ca or {}).get("count") != (cb or {}).get("count"):
            cardinality_ok = False
            first_div = first_div or ("cardinality", k)
        if ca != cb:
            identity_ok = False
            first_div = first_div or ("identity", k)
        if ra.get("graph", {}).get("mappings") != \
                rb.get("graph", {}).get("mappings"):
            mapping_ok = False
            first_div = first_div or ("mapping", k)
        if ra.get("action") != rb.get("action"):
            action_ok = False
            first_div = first_div or ("action", k)
    check("candidate cardinality consistent", cardinality_ok,
          first_div or "all equal")
    check("candidate identity consistent", identity_ok,
          first_div or "all equal")
    check("candidate-to-persistent mapping consistent", mapping_ok,
          first_div or "all equal")
    check("action trace consistent", action_ok, first_div or "all equal")
    root_a = chain_a[-1]["current_record_hash"] if chain_a else None
    root_b = chain_b[-1]["current_record_hash"] if chain_b else None
    check("hash chain roots equal", root_a == root_b,
          {"root_a": root_a, "root_b": root_b})
    return checks, root_a, root_b


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("verify")
    p.add_argument("--chain", required=True)
    p = sub.add_parser("compare")
    p.add_argument("--chain-a", required=True)
    p.add_argument("--chain-b", required=True)
    args = ap.parse_args()

    if args.cmd == "verify":
        try:
            chain = load_chain(args.chain)
        except (OSError, json.JSONDecodeError) as exc:
            print(json.dumps({"status": "ERROR", "error": str(exc)}))
            return 2
        ok, root, problems = verify_chain(chain)
        print(json.dumps({
            "status": "PASS" if ok else "FAIL",
            "prefix_count": len(chain),
            "chain_root": root,
            "genesis": GENESIS_HASH,
            "problems": problems[:20],
        }, indent=2))
        return 0 if ok else 1

    try:
        chain_a = load_chain(args.chain_a)
        chain_b = load_chain(args.chain_b)
    except (OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "ERROR", "error": str(exc)}))
        return 2
    oka, root_a, problems_a = verify_chain(chain_a)
    okb, root_b, problems_b = verify_chain(chain_b)
    checks, _, _ = compare_chains(chain_a, chain_b)
    status = "PASS" if (oka and okb and
                        all(c["pass"] for c in checks)) else "FAIL"
    print(json.dumps({
        "status": status,
        "chain_a_internally_valid": oka,
        "chain_b_internally_valid": okb,
        "chain_a_root": root_a,
        "chain_b_root": root_b,
        "checks": checks,
        "problems_a": problems_a[:10],
        "problems_b": problems_b[:10],
    }, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
