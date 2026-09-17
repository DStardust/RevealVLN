"""Read-only, stdlib graph-coverage diagnostic; no measured Qwen gradients."""
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
LINE = OUT.parents[2]
DATA = LINE/'reviews/Q35N_G1R_FAMILY_CERTIFICATION_V1'


def main():
    assert not (OUT/'TEMPORAL_CONTRACT_V2.json').exists()
    records = [json.loads(x) for x in (DATA/'SUPERVISION_ONLY.jsonl').read_text().splitlines()]
    rows = []
    for r in records:
        if r['continuation_trace_id'] != 'C0': continue
        path = DATA/'physical_traces'/f"1109_{r['history_id']}_C0.json"
        tr = json.loads(path.read_text()); t = r['prefix_cutoff_step']
        anchor = 'D' if r['task_id'] == 'g_T_v3' else 'K'
        positive_steps = [o['step'] for o in tr['observations'][:t+1] if o['see2'][anchor]]
        # Pixel-level exposure is distinct from completed two-frame SEE2.
        inventory = json.loads((DATA/'TASK_ROLE_INVENTORY.json').read_text())
        eligible = inventory['eligible'][anchor]
        visible_steps = [o['step'] for o in tr['observations'][:t+1]
                         if any(o['pixels'].get(str(i), 0) >= 256 for i in eligible)]
        row = {'task': r['task_id'], 'history': r['history_id'], 'cutoff': t,
               'prefix_decisions': t+1, 'first_see2': min(positive_steps) if positive_steps else None,
               'last_see2': max(positive_steps) if positive_steps else None,
               'last_threshold_visible_frame': max(visible_steps) if visible_steps else None,
               'trace_file_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'windows': []}
        for length in [2, 4, 8, 32, 64, 128, 207]:
            start = max(0, t-length+1)
            row['windows'].append({'tbptt_decisions': length, 'last_block_start': start,
                'anchor_see2_inside_final_graph': any(s >= start for s in positive_steps),
                'threshold_visible_frame_inside_final_graph': any(s >= start for s in visible_steps),
                'earliest_rgb_reencoded_in_final_block': max(0, start-1),
                'threshold_visible_rgb_available_to_final_block': any(s >= max(0, start-1) for s in visible_steps),
                'interpretation': 'graph reachability if memory detached immediately before final block; not measured gradients'})
        rows.append(row)
    report = {'decision': 'STATIC_TEMPORAL_GAP_RECORDED', 'rows': rows,
              'measured_long_history_gradient': None, 'measured_long_history_vram': None,
              'measured_query_encoder_runtime': None, 'scientific_pass': False,
              'caveat': 'Detachment prevents direct derivative to earlier activations, not every effect of shared parameters or all possible learning. Two-frame policy input re-encodes the frame immediately before block start; the separate available-RGB field accounts for this. Threshold absence does not exclude weaker visual cues.'}
    with (OUT/'TEMPORAL_CONTRACT_V2.json').open('x') as f: json.dump(report, f, indent=2)
    print(json.dumps([{'task': r['task'], 'history': r['history'], 'last_see2': r['last_see2'],
                       'last_threshold_visible_frame': r['last_threshold_visible_frame']} for r in rows], indent=2))


if __name__ == '__main__': main()
