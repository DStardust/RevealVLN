"""CPU-only evidence snapshot for the user-directed SR40 baseline reset."""
import collections,hashlib,json,math,subprocess,time,xml.etree.ElementTree as ET
from pathlib import Path
HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1];ROOT=LINE.parents[1]
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def main():
    assert not (HERE/'RESULT.json').exists()
    paths=dict(snapshot=LINE/'data_pipeline/ordinary_expansion_v1/training_snapshot_20260912_v1/run_001/RESULT.json',
        best=LINE/'closed_loop_bench/ordinary_expanded_dev_after_single_v1/run_001/RESULT.json',
        receipt=LINE/'sft_acceptance/ordinary_expanded_v1/formal/attempt_001/checkpoint_000004000.pt.json',
        epoch=LINE/'closed_loop_bench/ordinary_full_epoch_dev_v4/run_001/RESULT.json',
        full=LINE/'closed_loop_bench/r2r_ce_full_v2/run_001/RESULT.json',
        protocol=LINE/'closed_loop_bench/r2r_ce_full_v2/PROTOCOL.json',
        episodes=LINE/'closed_loop_bench/r2r_ce_full_v2/EPISODES_PRIVILEGED.json',
        model_config=LINE/'runtime/models/Qwen3.5-2B_15852e8/config.json',
        model=LINE/'sft_acceptance/ordinary_sync_recovery_v1/model.py',
        v12=LINE/'sft_acceptance/ordinary_stop_row_v12/FEATURE_PARITY.json',
        prefix=LINE/'reviews/Q35N_STOP_FEATURE_PREFIX_DIAGNOSTIC_V2/RESULT.json',
        prefix_failure=LINE/'reviews/Q35N_STOP_FEATURE_PREFIX_DIAGNOSTIC_V2/LAUNCH_FAILURE.json',
        process=LINE/'reviews/Q35N_STOP_FEATURE_PREFIX_DIAGNOSTIC_V2/PROCESS.json',
        review=LINE/'reports/TEACHER_INNOVATION_ADJUDICATION_20260911_V2_ZH.md')
    s=read(paths['snapshot']);b=read(paths['best']);e=read(paths['epoch']);f=read(paths['full']);p=read(paths['protocol']);eps=read(paths['episodes']);cfg=read(paths['model_config'])['text_config'];prefix=read(paths['prefix']);proc=read(paths['process'])
    assert s['counts']==dict(strict_routes=25415,instruction_records=37114,instruction_conditioned_decisions=2650347,unique_route_decisions=1720480)
    assert b['completed']==e['completed']==100 and b['sr']==.21 and e['sr']==.14
    assert f['completed']==len(eps)==p['episode_count']==1839 and p['house_count']==11
    assert p['success_distance']==3 and p['max_steps']==500 and p['optimizer_updates']==0
    assert cfg['num_hidden_layers']==24 and collections.Counter(cfg['layer_types'])==dict(linear_attention=18,full_attention=6)
    assert prefix['before']['max_abs_old']==prefix['after']['max_abs_old']==0 and prefix['forward_decisions']==160
    assert 'OWNED_IDENTITY_CHANGED' in read(paths['prefix_failure'])['error']
    xml=ET.fromstring(subprocess.check_output(['nvidia-smi','-q','-x'],text=True));gpu=next(x for x in xml.findall('gpu') if x.findtext('uuid')==p['gpu_uuid'])
    gpu_pids=[int(x.findtext('pid')) for x in gpu.findall('processes/process_info')]
    process_gone=not (Path('/proc')/str(proc['pid'])).exists()
    assert process_gone and gpu_pids==[]
    assert not (LINE/'sft_acceptance/ordinary_stop_row_v12/PROBE_RESULT.json').exists()
    out=dict(status='EVIDENCE_AND_ROADMAP_AUDITED',unix=time.time(),user_goal='ordinary_navigation_full_val_unseen_SR_at_least_40_percent',
      pass_goal=False,stage='BASELINE_RESET_PREPARATION',official_episode_count=1839,minimum_successes_for_40=math.ceil(.4*1839),
      snapshot=s['counts'],fit_houses=s['houses'],best_internal=dict(sr=b['sr'],spl=b['spl'],ndtw=b['ndtw'],newstage_decisions=read(paths['receipt'])['global_decisions']),
      full_epoch_internal=dict(sr=e['sr'],spl=e['spl'],ndtw=e['ndtw']),historical_full=dict(checkpoint_updates=f['checkpoint_updates'],sr=f['sr'],spl=f['spl'],selected_batch_size=f.get('selected_batch_size')),
      model_layers=dict(collections.Counter(cfg['layer_types'])),prior_stop_fit_started=False,
      prefix_diagnostic=dict(numerical_32_pass=True,launcher_failure_retained=True,unknown_child_exit_code=None,current_process_gone=process_gone,current_gpu1_empty=True,no_signal_sent_by_audit=True),
      old_goal='SUPERSEDED_BY_USER_NOT_ACHIEVED',new_gpu_launches=0,new_training_updates=0,new_simulator_actions=0,
      next_action='Pin and inspect ordinary-history/adapter recipe; fix common execution and exit-race tests before bounded new training',
      files={str(v):sha(v) for v in paths.values()},roadmap_sha256=sha(HERE/'ROADMAP_ZH.md'),audit_sha256=sha(Path(__file__)))
    with (HERE/'RESULT.json').open('x') as stream:json.dump(out,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in out.items() if k!='files'},ensure_ascii=False))
if __name__=='__main__':main()
