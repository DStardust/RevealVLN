"""Main-authorized family dependency closure recovery, distinct from store PASS.

V1 strict whole-store verdict remains preserved, including unrelated partials.
V2 may accept a family's fully verified dependency closure only if every retained
partial filename AND target pixel digest is absent from its whole evidence graph.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
path=HERE/'recovery.py'
if hashlib.sha256(path.read_bytes()).hexdigest()!='a091d6d29e0c91c573aaea9771a5f44207f23a0f41fbff4899ef11d739009011':
    raise ValueError('RECOVERY_V1_SOURCE_CHANGED')
spec=importlib.util.spec_from_file_location('required_content_recovery_v1',path)
v1=importlib.util.module_from_spec(spec);spec.loader.exec_module(v1)
require=v1.require
PARTIAL=re.compile(r'\.([0-9a-f]{64})\.(rgb|semantic)\.npy\.[0-9a-f]{32}\.partial')

def pixel_refs(document):
    result=set();queue=[document]
    while queue:
        value=queue.pop()
        if isinstance(value,dict):
            for key,item in value.items():
                if key in ('rgb_hash','semantic_hash','rgb_ref','semantic_ref'):
                    require(isinstance(item,str),'PIXEL_REFERENCE_TYPE')
                    raw=item.removeprefix('sha256:');require(re.fullmatch('[0-9a-f]{64}',raw),'PIXEL_REFERENCE_FORMAT')
                    result.add(raw+'.'+('semantic' if key.startswith('semantic') else 'rgb'))
                queue.append(item)
        elif isinstance(value,list):queue.extend(value)
    return result

def unrelated_partials(inventory,required_keys,evidence_blobs):
    require(inventory['actual_bytes_all_entries']<=inventory['limit_bytes'],'STORE_SIZE_LIMIT')
    by_name={x['name']:x for x in inventory['files']};excluded=[]
    for problem in inventory['problems']:
        name=problem['name'];match=PARTIAL.fullmatch(name)
        require(problem['error']=='FAILURE_PARTIAL_OR_FOREIGN_ENTRY' and match is not None,'NONPARTIAL_STORE_ERROR')
        require(name in by_name,'PARTIAL_NOT_FULLY_INVENTORIED')
        digest,kind=match.groups()
        require(digest+'.'+kind not in required_keys,'PARTIAL_TARGET_IN_REQUIRED_CONTENT_INDEX')
        require(all(digest.encode() not in blob and name.encode() not in blob for blob in evidence_blobs),'PARTIAL_OR_TARGET_REFERENCED_IN_EVIDENCE_GRAPH')
        excluded.append(dict(by_name[name],target_pixel_sha256=digest,target_kind=kind,
            proven_absent_from_family_evidence_graph=True,retained_in_original_store=True))
    return excluded

def dependency_closure(folder,reader,inventory):
    root=folder/'export_v4';index=reader.json(root/'CONTENT_INDEX.json');required=set(index)
    # Include all actual bundle traces (discovery too), metadata, exported history,
    # policy, query and supervision files. No selection of only favorable records.
    paths=sorted(set(root.rglob('*'))|set(folder.glob('*.json'))|set((folder/'traces').glob('*.json')))
    require(all(p.absolute()==p.resolve() for p in paths),'EVIDENCE_GRAPH_SYMLINK')
    paths=[p for p in paths if not p.is_dir()]
    require(len(paths)<=512,'EVIDENCE_GRAPH_FILE_CAP')
    blobs=[];refs=set()
    for path in paths:
        raw=reader.read(path);blobs.append(raw)
        # Export holds the canonical whole history+continuation traces. V1 proved
        # all 27 actual cert traces have exactly that same observation content.
        # Discovery-only traces are additionally searched for partial references,
        # but unused scout/component frames are not falsely called training refs.
        if path.suffix=='.json':
            parsed=json.loads(raw)
            if path.is_relative_to(root):refs.update(pixel_refs(parsed))
        elif path.suffix=='.jsonl':
            for line in raw.splitlines():
                parsed=json.loads(line)
                if path.is_relative_to(root):refs.update(pixel_refs(parsed))
    require(refs<=required,'UNINDEXED_PIXEL_REFERENCE_IN_FAMILY_GRAPH')
    indexed=[]
    for key,item in index.items():
        require(key==item['raw_pixel_sha256']+'.'+item['kind'],'CONTENT_KEY_IDENTITY')
        path=v1.safe(v1.LINE/item['line_relative_path'])
        require(path.parent==folder.parent.parent/'content','CONTENT_OUTSIDE_SOURCE_STORE')
        raw=reader.read(path)
        require(hashlib.sha256(raw).hexdigest()==item['file_sha256'],'REQUIRED_BLOB_FILE_HASH')
        v1.sem.loader_module.npy_pixels(raw,item['raw_pixel_sha256'],item['kind'])
        indexed.append({'key':key,'path':str(path),'sha256':item['file_sha256'],'bytes':len(raw)})
    excluded=unrelated_partials(inventory,required,blobs)
    return {'type':'FAMILY_REQUIRED_CONTENT_CLOSURE_RESEAL_V2','required_content_pass':True,
        'evidence_graph_files':list(map(str,paths)),'required_blob_count':len(indexed),'required_blobs':indexed,
        'referenced_pixel_key_count':len(refs),'all_graph_pixel_refs_in_content_index':True,
        'excluded_uncommitted_partials':excluded,'whole_store_inventory_pass':inventory['content_inventory_pass'],
        'whole_store_bytes_including_excluded_partials':inventory['actual_bytes_all_entries'],
        'original_store_close_fabricated':False,'original_batch_pass':False,
        'required_graph_scope':'complete export history/query/policy/M2/normalization refs; 27 cert traces independently proven content-identical by V1 semantics',
        'partial_exclusion_graph_scope':'all bundle discovery/certification traces and complete export plus bundle metadata'}

def audit_family_scope(run_root,ident,output):
    out=v1.safe(output);require(out.is_relative_to(HERE) and out!=HERE,'OUTPUT_SCOPE');out.mkdir(parents=True,exist_ok=False)
    result={'grade':'RECOVERY_REQUIRED_CONTENT_INSUFFICIENT','recovery_content_pass':False,'quality_pass':False,
        'training_admission':False,'requires_main_training_admission':True,'original_batch_pass':False,
        'scientific_pass':False,'model_gain_pass':False,'candidate_id':ident,'run_root':str(run_root),'errors':[]}
    reader=v1.Reader()
    try:
        strict=v1.audit_recovery(run_root,ident,out/'strict_full_store_audit')
        require(strict['recovery_content_pass'] is True or strict['errors']==['ValueError: READ_ONLY_STORE_RESEAL_FAILED'],'STRICT_FAILURE_NOT_ONLY_STORE_INVENTORY')
        require(strict.get('semantics',{}).get('semantic_content_pass') is True,'SEMANTIC_EVIDENCE_MISSING')
        require(strict.get('resource_evidence',{}).get('all_recorded_resource_samples_pass') is True,'RECORDED_RESOURCE_EVIDENCE_MISSING')
        # The V1 failed path deliberately does not claim a final bookend recheck.
        # Re-establish that recheck here over every source file V1 actually read.
        seals=reader.json(out/'strict_full_store_audit/RECOVERY_SOURCE_SEALS.json')
        for path,expected in seals.items():require(reader.read(path,False)==expected,'SOURCE_CHANGED_AFTER_STRICT_AUDIT')
        inventory=reader.json(out/'strict_full_store_audit/RECOVERY_CONTENT_RESEAL.json')
        run=v1.safe(run_root);folder=run/'bundles'/ident
        closure=dependency_closure(folder,reader,inventory)
        reader.verify_again()
        v1.save(out/'FAMILY_REQUIRED_CONTENT_RESEAL.json',closure)
        result.update(grade='RECOVERED_FAMILY_REQUIRED_CONTENT_VERIFIED_RUNTIME_CENSORED',recovery_content_pass=True,
            source_and_phase_binding_verified=True,preaction_configuration_verified=True,semantics=strict['semantics'],
            committed_prefix=strict['committed_prefix'],family_completion=strict['family_completion'],resource_evidence=strict['resource_evidence'],
            whole_store_inventory_pass=inventory['content_inventory_pass'],original_store_closed_pass=False,
            excluded_uncommitted_partials=closure['excluded_uncommitted_partials'],required_blob_count=closure['required_blob_count'],
            required_main_decision='admit separately graded recovered required-content family to FIT training with provenance; original failed batch/store remain failed',
            language_scope='controlled templates only; no natural language transfer or model gain',
            scope='family dependency closure, not whole-store or original-run completion')
    except (ValueError,KeyError,OSError,TypeError,AssertionError) as e:result['errors'].append(type(e).__name__+': '+str(e))
    result['auditor_sources']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE/'family_scope_v2.py',HERE/'recovery.py',HERE/'semantics.py')}
    result['read_bytes_counted']=reader.bytes_read
    v1.save(out/'REQUIRED_CONTENT_SOURCE_SEALS.json',reader.seals);v1.save(out/'REPORT.json',result);return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--bundle',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(audit_family_scope(a.run,a.bundle,a.output),indent=2))
