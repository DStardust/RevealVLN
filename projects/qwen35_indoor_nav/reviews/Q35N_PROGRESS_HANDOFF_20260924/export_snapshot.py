"""One-shot read-only publication; no model invocation or service mutation.

Run in a NEW output directory. Pair seals are selected at the initial cutoff;
later completions are deliberately not added to this publication.
"""
import argparse, csv, datetime, gzip, hashlib, io, json, os, subprocess, tarfile, time
from collections import Counter
from pathlib import Path


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def main(root, out):
    root = root.resolve(); out = out.resolve()
    line = root / 'projects/qwen35_indoor_nav'
    cutoff_ns = time.time_ns()
    utc = datetime.datetime.fromtimestamp(cutoff_ns/1e9, datetime.timezone.utc).isoformat()
    assert not (out/'SNAPSHOT.json').exists(), 'REFUSE_OVERWRITE_SNAPSHOT'
    out.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=root, text=True).strip()
    definitions = [
        ('V20', line/'sft_acceptance/ordinary_stop_contrast_v20', 'contrast_001', ['DEV', 'UNSEEN']),
        ('FINAL100K', line/'closed_loop_bench/ordinary_long100k_transfer_v1', 'transfer_001', ['UNSEEN']),
    ]
    exposure = json.loads((definitions[1][1]/'EXPOSURE_AUDIT.json').read_text())['public_unseen_houses']
    selected = []; metadata = set(); phase_items = []; attempts = []
    # Only explicit run/session/pair depths; never traverse RGB frame trees.
    for name, folder, run_id, phases in definitions:
        run = folder/'runs'/run_id
        for directory in [folder, folder/'ops_recovery_r1', run]:
            if directory.exists():
                metadata.update(p for p in directory.iterdir() if p.is_file() and p.suffix in ('.json','.jsonl','.md','.txt','.log','.diff'))
        metadata.update(p for p in (folder/'standalone_jobs').glob('*/*') if p.is_file() and p.suffix in ('.json','.log'))
        for phase in phases:
            directory = run/phase
            if not directory.exists():
                continue
            metadata.update(p for p in directory.iterdir() if p.is_file())
            sessions = sorted((directory/'sessions').iterdir())
            configs = [json.loads((s/'CONFIG.json').read_text()) for s in sessions]
            config = configs[0]
            orderpath = Path(config['order_path']); episodespath = Path(config['episodes_path'])
            order = json.loads(orderpath.read_text())
            assert len(order) == config['planned_pairs']
            metadata.update([orderpath, episodespath])
            phase_selected = []; partial = []
            for session, cfg in zip(sessions, configs):
                assert cfg['planned_pairs'] == len(order)
                assert cfg['candidate_sha256'] == config['candidate_sha256']
                metadata.update(p for p in session.iterdir() if p.is_file() and p.suffix in ('.json','.jsonl','.log'))
                metadata.update(p for p in (session/'attempts').glob('*/*') if p.is_file() and p.suffix in ('.json','.jsonl','.log'))
                for pd in sorted((session/'pairs').iterdir()):
                    if not pd.is_dir(): continue
                    seal = pd/'PAIR.json'
                    if seal.exists() and seal.stat().st_mtime_ns <= cutoff_ns:
                        raw = seal.read_bytes(); value = json.loads(raw)
                        item = dict(experiment=name, phase=phase, path=seal, raw=raw, value=value)
                        selected.append(item); phase_selected.append(item)
                    else:
                        partial.append(dict(path=str(pd.relative_to(root)), seal_at_cutoff=False,
                                            status='NO_COMPLETE_PAIR_SEAL_AT_CUTOFF'))
            phase_items.append((name, phase, order, phase_selected, config))
            attempts.extend(dict(experiment=name, phase=phase, **p) for p in partial)
    print(json.dumps({'cutoff_utc':utc,'selected_pairs':len(selected)},ensure_ascii=False),flush=True)
    # Freeze lightweight status/attempt evidence. These files have individual capture times.
    copied = []
    for source in sorted(metadata):
        if source.stat().st_size > 16*1024*1024:
            copied.append(dict(source_path=str(source.relative_to(root)), status='OMITTED_LARGE_RUNTIME_LOG',bytes=source.stat().st_size));continue
        raw = source.read_bytes(); rel = source.relative_to(root)
        dest = out/'captured'/rel; dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open('xb') as f:f.write(raw)
        copied.append(dict(source_path=str(rel),snapshot_path=str(dest.relative_to(out)),bytes=len(raw),sha256=digest(raw),captured_unix=time.time()))
    save(out/'CAPTURE_INDEX.json',copied)
    save(out/'INCOMPLETE_PAIR_ATTEMPTS.json',attempts)

    # Complete original JSON/log traces, without images or feature tensors. Raw chunk
    # size is bounded so even incompressible archives fit GitHub's 100 MB limit.
    archive_dir = out/'traces'; archive_dir.mkdir(exist_ok=True)
    archive = None; compressed = None; backing = None; raw_bytes = 0; chunk = 0
    trace_index = []; chunks = []
    def close_archive():
        nonlocal archive, compressed, backing
        if archive is not None:
            archive.close(); compressed.close(); backing.close()
            path = archive_dir/f'part_{chunk:03d}.tar.gz'
            chunks.append(dict(path=str(path.relative_to(out)),bytes=path.stat().st_size,sha256=digest(path.read_bytes())))
            archive = compressed = backing = None
    def add_raw(rel, raw):
        nonlocal archive, compressed, backing, raw_bytes, chunk
        if archive is None or raw_bytes+len(raw)+2048 > 48*1024*1024:
            close_archive(); chunk += 1; raw_bytes = 0
            backing = (archive_dir/f'part_{chunk:03d}.tar.gz').open('xb')
            compressed = gzip.GzipFile(fileobj=backing,mode='wb',compresslevel=3,mtime=0)
            archive = tarfile.open(fileobj=compressed,mode='w|')
        info = tarfile.TarInfo(rel); info.size=len(raw); info.mtime=0; info.mode=0o644
        archive.addfile(info,io.BytesIO(raw));raw_bytes+=len(raw)+2048
        return f'traces/part_{chunk:03d}.tar.gz'
    pair_index = []
    for number,item in enumerate(selected,1):
        p=item['path'];value=item['value']; rel=str(p.relative_to(root))
        arc=add_raw(rel,item['raw'])
        trace_index.append(dict(path=rel,bytes=len(item['raw']),sha256=digest(item['raw']),archive=arc,kind='PAIR_SEAL'))
        for name,expected in sorted(value['trace_hashes'].items()):
            source=p.parent/name; raw=source.read_bytes()
            assert digest(raw)==expected, 'SEALED_TRACE_CHANGED:'+str(source)
            reltrace=str(source.relative_to(root));arc=add_raw(reltrace,raw)
            trace_index.append(dict(path=reltrace,bytes=len(raw),sha256=expected,archive=arc,kind='SEALED_TRACE'))
        assert p.read_bytes()==item['raw'], 'PAIR_SEAL_CHANGED_DURING_EXPORT'
        pair_index.append(dict(experiment=item['experiment'],phase=item['phase'],rank=value['rank'],episode_id=value['episode_id'],source_path=rel,sha256=digest(item['raw'])))
        if number%250==0:print('ARCHIVED_SEALED_PAIRS',number,flush=True)
    close_archive()
    save(out/'TRACE_INDEX.json',trace_index)
    save(out/'PAIR_INDEX.json',pair_index)
    save(out/'TRACE_ARCHIVES.json',chunks)

    rows=[];summaries=[]
    fields=['experiment','phase','rank','episode_id','house','inference_order','status','A_success','B_success','delta_success','outcome','A_spl','B_spl','A_ndtw','B_ndtw','A_osr','B_osr','A_steps','B_steps','A_collisions','B_collisions','A_termination','B_termination','input_prefix_matched','base_action_prefix_matched','logits_bitwise_equal','max_base_logit_delta','state_unchanged','first_action_divergence','source_path','pair_sha256','exposure_stratum']
    for name,phase,order,items,config in phase_items:
        sealed={i['value']['rank']:i for i in items}
        assert len(sealed)==len(items),'DUPLICATE_COMPLETE_RANK'
        phase_rows=[]
        for planned in order:
            rank=planned['rank']; item=sealed.get(rank)
            row=dict(experiment=name,phase=phase,rank=rank,episode_id=planned['episode_id'],house=planned['house'],inference_order='/'.join(planned['inference_order']),status='NOT_COMPLETE_AT_SNAPSHOT')
            row['exposure_stratum']=exposure.get(planned['house'],{}).get('stratum','INTERNAL_DEV_TRAIN_SPLIT') if name=='FINAL100K' else ('PUBLIC_PREVIOUSLY_EXPOSED' if phase=='UNSEEN' else 'INTERNAL_DEV_TRAIN_SPLIT')
            if item:
                value=item['value'];assert value['episode_id']==planned['episode_id'] and value['house']==planned['house']
                row.update(status='COMPLETE_PAIR_SEALED',source_path=str(item['path'].relative_to(root)),pair_sha256=digest(item['raw']))
                for arm in ('A','B'):
                    for metric in ('success','spl','ndtw','steps','collisions','termination'):
                        row[arm+'_'+metric]=value[arm][metric]
                    row[arm+'_osr']=value[arm]['oracle_success']
                row['delta_success']=row['B_success']-row['A_success']
                row['outcome']='WIN' if row['delta_success']>0 else 'LOSS' if row['delta_success']<0 else 'RETAINED_SUCCESS' if row['A_success'] else 'BOTH_FAILURE'
                for key in ('input_prefix_matched','base_action_prefix_matched','logits_bitwise_equal','max_base_logit_delta','state_unchanged','first_action_divergence'):row[key]=value[key]
            phase_rows.append(row);rows.append(row)
        def summarize_subset(rs):
            complete=[x for x in rs if x['status']=='COMPLETE_PAIR_SEALED'];n=len(complete);planned=len(rs);missing=planned-n
            result=dict(planned_pairs=planned,complete_pairs=n,pending_pairs=missing,
                        outcome_counts=dict(Counter(x['outcome'] for x in complete)),
                        win_episode_ids=[x['episode_id'] for x in complete if x['outcome']=='WIN'],
                        loss_episode_ids=[x['episode_id'] for x in complete if x['outcome']=='LOSS'])
            for arm in ('A','B'):
                successes=sum(x[arm+'_success'] for x in complete)
                result[arm]=dict(successes=successes,**{metric+'_on_completed':sum(x[arm+'_'+metric] for x in complete)/n if n else None for metric in ('success','spl','ndtw','osr','steps','collisions')},sr_identification_bounds=[successes/planned,(successes+missing)/planned])
            delta=result['B']['successes']-result['A']['successes']
            result['delta_sr_on_completed']=delta/n if n else None
            result['full_denominator_delta_identification_bounds']=[(delta-missing)/planned,(delta+missing)/planned]
            result['bounds_are_not_confidence_intervals']=True
            result['recorded_pair_audit']={key:sum(x[key] is True for x in complete) for key in ('input_prefix_matched','base_action_prefix_matched','logits_bitwise_equal','state_unchanged')}
            result['max_base_logit_delta']=max((x['max_base_logit_delta'] for x in complete),default=None)
            return result
        summary=dict(experiment=name,phase=phase,**summarize_subset(phase_rows))
        summary['houses']={h:summarize_subset([x for x in phase_rows if x['house']==h]) for h in sorted(set(x['house'] for x in phase_rows))}
        summary['exposure_strata']={s:summarize_subset([x for x in phase_rows if x['exposure_stratum']==s]) for s in sorted(set(x['exposure_stratum'] for x in phase_rows))}
        summary['reference_sha256']=config['reference_checkpoint_sha256'];summary['candidate_sha256']=config['candidate_sha256']
        summaries.append(summary)
    with (out/'ROLLOUTS.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    snapshot=dict(status='PUBLICATION_SNAPSHOT_NOT_FINAL_EVALUATION',cutoff_utc=utc,cutoff_unix_ns=cutoff_ns,
                  metadata_capture='Individual capture timestamps in CAPTURE_INDEX; live files may lag immutable pair seals.',
                  source_commit=commit,branch=branch,complete_pairs=len(selected),planned_pair_slots=len(rows),summaries=summaries,
                  validation='Original sealed trace SHA256 checked during export. Recorded prefix/state flags are copied, not a new independent model/metric rerun.',
                  scientific_scope='Public exposed evaluation; final100k includes declared training-house overlaps. No blind-generalization or adoption claim.',
                  omissions=['Raw RGB and scene/semantic assets','312 MB V20 FEATURES.pt','Base checkpoint/processor environment already indexed by original source locks','In-flight unsealed full traces, retained locally; partial attempt directories indexed here'],
                  background_policy='No service stopped, signaled, retrained, or polled continuously by this exporter. Existing autonomous pipelines and web monitor continue.')
    save(out/'SNAPSHOT.json',snapshot)
    print(json.dumps({'status':'SNAPSHOT_EXPORTED','pairs':len(selected),'trace_files':len(trace_index),'archive_bytes':sum(x['bytes'] for x in chunks),'phases':[{k:s[k] for k in ('experiment','phase','complete_pairs','planned_pairs','delta_sr_on_completed')} for s in summaries]},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();main(args.root,args.output)
