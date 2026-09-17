"""Read-only forensic analysis; new outputs only in this diagnosis directory."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

OUT = Path(__file__).resolve().parent
RUNTIME = OUT.parents[1]
RUN = RUNTIME / "feedback_generation_v1/run_v1"

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")

def describe(values):
    ordered=sorted(values)
    return {"count":len(values), "sum":sum(values), "mean":statistics.mean(values) if values else None,
            "median":statistics.median(values) if values else None,
            "p95":ordered[min(len(ordered)-1, int(len(ordered)*.95))] if values else None}

def main():
    result=json.loads((RUN/"result.json").read_text())
    budget=json.loads((RUN/"BUDGET_FINAL.json").read_text())
    records=[]; raw_bytes=0; previous="0"*64; head_bytes=0
    for seq,line in enumerate((RUN/"journal/events.jsonl").open("rb")):
        record=json.loads(line)
        assert canonical(record)+b"\n"==line
        body={k:v for k,v in record.items() if k!="hash"}
        assert record["seq"]==seq and record["prev"]==previous
        assert record["hash"]==hashlib.sha256(canonical(body)).hexdigest()
        previous=record["hash"];raw_bytes+=len(line);records.append(record)
        if seq==0:config_hash=hashlib.sha256(canonical(record["payload"])).hexdigest()
        head_bytes+=len(canonical({"version":1,"count":seq+1,"byte_length":raw_bytes,"last_hash":previous,"config_hash":config_hash}))
    head=json.loads((RUN/"journal/HEAD.json").read_text())
    assert head["count"]==len(records) and head["byte_length"]==raw_bytes and head["last_hash"]==previous
    kinds=Counter(r["kind"] for r in records)
    budget_records=[r for r in records if r["kind"]=="budget"]
    actions=[(i,r) for i,r in enumerate(records) if r["kind"]=="action_completed"]
    gaps={"replay_after_action_checkpoint_to_observe_entry":[],"feedback_after_observe_checkpoint_to_loop_entry":[]}
    gap_exclusions=Counter()
    for (a,prev),(b,next_) in zip(actions,actions[1:]):
        segment=records[a+1:b]
        if len(segment)!=5 or any(r["kind"]!="budget" for r in segment):
            gap_exclusions["not_exact_five_budget_records"]+=1;continue
        prev_value=prev["payload"]["value"];next_value=next_["payload"]["value"]
        phase=prev_value.get("phase","replay")
        if prev["payload"]["bundle"]!=next_["payload"]["bundle"] or phase!=next_value.get("phase","replay"):
            gap_exclusions["boundary_or_phase_change"]+=1;continue
        times=[r["payload"]["last_clock"] for r in segment]
        # Source-mapped intervals include one synchronous budget-journal commit
        # plus adjacent Python checks; they are NOT direct fsync syscall timings.
        key="feedback_after_observe_checkpoint_to_loop_entry" if phase=="feedback_discovery" else "replay_after_action_checkpoint_to_observe_entry"
        offset=1 if phase=="feedback_discovery" else 0
        delta=times[offset+1]-times[offset]
        assert delta>=0
        gaps[key].append(delta)
    trace_stats=[]; totals=Counter();unique={"rgb":set(),"semantic":set()}
    for folder in sorted((RUN/"bundles").iterdir()):
        summary=json.loads((folder/"result.json").read_text())
        data=Counter();modes=Counter();failures=Counter()
        for path in sorted((folder/"traces").glob("*.json")):
            tr=json.loads(path.read_text());data["traces"]+=1;data["actions"]+=len(tr["actions"])
            data["observations"]+=len(tr["observations"]);data["trace_json_bytes"]+=path.stat().st_size
            modes[tr.get("discovery_kind","replay")]+=1
            failures[str(tr.get("failure_reason"))]+=1
            for obs in tr["observations"]:
                for kind in unique:unique[kind].add(obs[kind+"_hash"])
        phase=budget["bundles"][folder.name]["discovery"]
        data["phase_seconds"]=phase["finished"]-phase["started"]
        totals.update(data)
        trace_stats.append(dict(summary,**data,modes=dict(modes),failure_reasons=dict(failures)))
    content=Counter();content_bytes=Counter()
    for path in (RUN/"content").iterdir():
        assert path.is_file() and not path.is_symlink()
        kind=path.name.split(".")[-2];content[kind]+=1;content_bytes[kind]+=path.stat().st_size
    samples=[json.loads(x) for x in (RUN/"RESOURCE_SAMPLES.jsonl").read_text().splitlines()]
    phase_seconds=sum(x["phase_seconds"] for x in trace_stats)
    info={"scope":"OFFLINE_READ_ONLY_DIAGNOSIS_NO_GPU_NO_NEW_BENCHMARK", "source_run":str(RUN),
          "wall_seconds":result["wall_seconds"],"phase_seconds":phase_seconds,
          "wall_minus_discovery_phases_seconds":result["wall_seconds"]-phase_seconds,
          "source_counts":result["counts"],"trace_totals":dict(totals),"by_bundle":trace_stats,
          "journal":{"chain_and_HEAD_pass":True,"records":len(records),"kinds":dict(kinds),
                     "event_file_bytes":raw_bytes,"derived_cumulative_HEAD_bytes_written":head_bytes,
                     "code_implied_successful_fsync_calls":3*len(records)+1,
                     "fsync_count_is_source_derived_not_syscall_measured":True},
          "content":{"file_counts":dict(content),"bytes":dict(content_bytes),"total_bytes":sum(content_bytes.values()),
                     "referenced_unique_hashes":{k:len(v) for k,v in unique.items()},
                     "code_implied_successful_commit_fsync_calls":3*sum(content.values()),
                     "put_calls_lower_bound_from_saved_observations":2*totals["observations"],
                     "full_payload_final_audits_in_worker":2},
          "narrow_source_mapped_checkpoint_intervals":{k:describe(v) for k,v in gaps.items()},
          "interval_exclusions":dict(gap_exclusions),
          "resource_samples":{"count":len(samples),"gpu_utilization":describe([s["utilization"] for s in samples]),
                              "disk_scan_samples":sum("disk_bytes" in s for s in samples),
                              "graphics_process_inventory_complete":False},
          "not_measured":["total fsync syscall time","renderer total time","full store total time","planner total time", "final audit exact time"]}
    with (OUT/"METRICS.json").open("x") as stream:json.dump(info,stream,indent=2)
    print(json.dumps(info,indent=2))

if __name__=="__main__":main()
