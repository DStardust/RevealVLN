"""Train-house-only semantic proposals; no renderer, model, or training imports.

MP3D ASCII 1.1 fields match the project-pinned Habitat v0.1.7 C++ loader.
No coordinate transforms or visibility/reachability conclusions are made here.
All outputs are supervision-side candidate metadata, never policy inputs.
"""
import collections
import gzip
import hashlib
import heapq
import itertools
import json
import math
from pathlib import Path

ROOT = Path('/mnt/data_nas/deeprobotics/daiyang/vla')
HERE = Path(__file__).resolve().parent
LINE = HERE.parents[1]
ROOMS = dict(zip(
    ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','r','s','t','u','v','w','x','y','z','B','C','D','S','Z'],
    ['bathroom','bedroom','closet','dining room','entryway/foyer/lobby','familyroom/lounge','garage','hallway','library','laundryroom/mudroom','kitchen','living room','meetingroom/conferenceroom','lounge','office','porch/terrace/deck','rec/game','stairs','toilet','utilityroom/toolroom','tv','workout/gym/exercise','outdoor','balcony','other room','bar','classroom','dining booth','spa/sauna','junk']))
INDOOR_ROOMS = {'bathroom','bedroom','dining room','familyroom/lounge','hallway','kitchen','living room','office','lounge','toilet','tv'}
# Finite, declared vocabulary. Unknown/raw compound labels are not guessed.
RAW = {'bed': {'bed'}, 'sink': {'sink'}, 'tv_monitor': {'tv','television'},
       'chair': {'chair','dining chair','armchair','office chair'},
       'sofa': {'sofa','couch'}, 'table': {'table','dining table','coffee table'},
       'toilet': {'toilet'}, 'plant': {'plant'}}
RESERVED = {0,65535,4294967295}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def scoped(path):
    path = Path(path).resolve(strict=True)
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError('OUT_OF_SCOPE_OR_MISSING')
    return path


def parse_house(text):
    lines = text.splitlines()
    if not lines or lines[0].strip() != 'ASCII 1.1':
        raise ValueError('UNSUPPORTED_HOUSE_VERSION')
    regions, categories, objects = {}, {}, {}
    for line in lines[1:]:
        t = line.split()
        if not t or t[0] not in {'R','C','O'}:
            continue
        if t[0] == 'R':
            key = int(t[1])
            if key in regions:
                raise ValueError('DUPLICATE_REGION')
            regions[key] = {'level': int(t[2]), 'room': ROOMS.get(t[5])}
        elif t[0] == 'C':
            key = int(t[1])
            if key in categories:
                raise ValueError('DUPLICATE_CATEGORY')
            categories[key] = {'raw': t[3].replace('#',' '), 'mpcat40': t[5]}
        else:
            key, region, category = map(int,t[1:4])
            if key in objects:
                raise ValueError('DUPLICATE_OBJECT')
            center = list(map(float,t[4:7]))
            if len(center) != 3 or not all(math.isfinite(v) for v in center):
                raise ValueError('INVALID_CENTER')
            if region >= 0 and region not in regions or category >= 0 and category not in categories:
                raise ValueError('DANGLING_SEMANTIC_REFERENCE')
            objects[key] = {**categories.get(category, {'raw':'','mpcat40':''}),
                            'room': regions.get(region,{}).get('room'),
                            'region_index': region, 'object_index': key,
                            'level': regions.get(region,{}).get('level'),
                            'house_frame_center': center}
    return objects


def semantic_groups(objects):
    grouped = collections.defaultdict(list)
    for key, obj in sorted(objects.items()):
        raw = ' '.join(obj['raw'].lower().split())
        if key in RESERVED or obj['room'] not in INDOOR_ROOMS or raw not in RAW.get(obj['mpcat40'],set()):
            continue
        grouped[(obj['mpcat40'],obj['room'],raw)].append(obj)
    return grouped


def role_spec(signature):
    cat, room, raw = signature
    return {'mpcat40':cat, 'room':room, 'raw_match':{'mode':'exact','value':raw}}


def task_spec(roles):
    def wording(role):
        r = roles[role]
        return 'the '+r['raw_match']['value']+' in the '+r['room']
    return {task:{'anchor':role,'terminal':'terminal',
                  'instruction': 'First see '+wording(role)+' in two consecutive observations, then see '+wording('terminal')+' in two consecutive observations, and stop immediately.'}
            for task,role in [('task_A','anchor_A'),('task_B','anchor_B')]}


def propose(house, objects, limit=32):
    if type(limit) is not int or not 1 <= limit <= 128:
        raise ValueError('BOUNDED_PROPOSAL_LIMIT')
    groups = semantic_groups(objects)
    keys = sorted(groups)
    selected, rejected, total = [], collections.Counter(), 0
    # Unordered anchors avoid counting A/B relabeling as a second data family.
    for a,b in itertools.combinations(keys,2):
        if a[1] == b[1]:
            rejected['anchor_room_not_distinct'] += 1
            continue
        for terminal in keys:
            if terminal in (a,b) or terminal[1] in (a[1],b[1]):
                continue
            if min(math.dist(x['house_frame_center'],y['house_frame_center']) for x in groups[a] for y in groups[b]) < 1.5:
                rejected['anchor_centers_too_close'] += 1
                continue
            for irrelevant in keys:
                if irrelevant in (a,b,terminal):
                    continue
                signatures = {'anchor_A':a,'anchor_B':b,'terminal':terminal,'irrelevant':irrelevant}
                # At least one shared level is needed for this first static-floor
                # producer, but a common level still does not prove reachability.
                levels = set.intersection(*[{o['level'] for o in groups[v]} for v in signatures.values()])
                if not levels:
                    rejected['no_shared_metadata_level'] += 1
                    continue
                roles = {k:role_spec(v) for k,v in signatures.items()}
                fingerprint = digest({'version':1,'house_id':house,'roles':roles})
                total += 1
                order = int(fingerprint,16)
                if len(selected) >= limit and order >= -selected[0][0]:
                    continue
                proposal = {'candidate_id':'MS1_'+fingerprint[:20],
                    'semantic_fingerprint_sha256':fingerprint,'house_id':house,
                    'quality_tier':'SEMANTIC_CANDIDATE','roles':roles,
                    'tasks':task_spec(roles),'expected_eligible':{k:[o['object_index'] for o in groups[v]] for k,v in signatures.items()},
                    'shared_metadata_levels':sorted(levels),'visibility_pass':None,
                    'physical_replay_pass':None,'matrix_pass':None,'shortcut_pass':None,
                    'training_admission':False}
                if len(selected) < limit:
                    heapq.heappush(selected,(-order,fingerprint,proposal))
                else:
                    heapq.heapreplace(selected,(-order,fingerprint,proposal))
    # Hash order avoids raw alphabetical preference and is independent of outcomes.
    proposals = sorted((x[2] for x in selected),key=lambda r:r['semantic_fingerprint_sha256'])
    return proposals, {'eligible_role_signatures':len(groups),
        'enumerated_semantic_combinations':total,
        'bounded_manifest_proposals':len(proposals),
        'prefilter_rejections':dict(sorted(rejected.items())),
        'trimmed_by_predeclared_limit':max(0,total-limit)}


def validate_quality(record):
    tiers = {'SEMANTIC_CANDIDATE': [],
             'PHYSICAL_REPLAY_CERTIFIED':['asset_identity','runtime_role_identity','observable_witnesses','legal_real_histories','matched_physical_join','replay_27','matrix_18','label_reversal','irrelevant_invariance','language_dependence','causal_export_readback'],
             'MECHANISM_TRAIN_READY':['asset_identity','runtime_role_identity','observable_witnesses','legal_real_histories','matched_physical_join','replay_27','matrix_18','label_reversal','irrelevant_invariance','language_dependence','causal_export_readback','length_shortcut_control','current_view_shortcut_control','query_only_shortcut_control','house_split_integrity','strong_state_target_available']}
    tier = record.get('quality_tier')
    if tier not in tiers:
        raise ValueError('UNKNOWN_QUALITY_TIER')
    checks = record.get('checks',{})
    if any(checks.get(name) is not True for name in tiers[tier]):
        raise ValueError('MISSING_QUALITY_EVIDENCE')
    if record.get('training_admission') is True and tier != 'MECHANISM_TRAIN_READY':
        raise ValueError('PREMATURE_TRAINING_ADMISSION')
    return True


def build(split_path):
    split_path = scoped(split_path)
    split = json.loads(split_path.read_text())
    # The caller passes the actual shared ordinary/mechanism split, not a new one.
    rows = split.get('houses')
    if rows is None:
        if not all(name in split for name in ('FIT','INTERNAL_DEV','INTERNAL_CONFIRM')):
            raise ValueError('SHARED_SPLIT_SCHEMA')
        groups = [split[name] for name in ('FIT','INTERNAL_DEV','INTERNAL_CONFIRM')]
        if len(set(sum(groups,[]))) != sum(map(len,groups)):
            raise ValueError('SHARED_SPLIT_OVERLAP')
        rows = [{'house_id':h,'split':name} for name in ('FIT','INTERNAL_DEV','INTERNAL_CONFIRM') for h in split[name]]
    if isinstance(rows,dict):
        rows = [{'house_id':k,'split':v if isinstance(v,str) else v['split']} for k,v in rows.items()]
    fit = sorted(r['house_id'] for r in rows if r['split'] in {'FIT','FIT_PILOT'})
    source = scoped(ROOT/'data/phase0/raw/r2r_vlnce_v1-3/train/train.json.gz')
    if sha(source) != 'f411066b53f96d1241c045fd05a6a9e01b484c2ed9369f5b6f41806969056a34':
        raise ValueError('SOURCE_CHANGED')
    with gzip.open(source,'rt') as f:
        episodes = json.load(f)['episodes']
    train_houses = {Path(ep['scene_id']).parent.name for ep in episodes}
    if len(fit) != len(set(fit)) or not set(fit) <= train_houses:
        raise ValueError('INVALID_FIT_HOUSES')
    manifests, coverage, reads, ledger = [], [], [], []
    for house in fit:
        path = scoped(ROOT/'third_party/ETP-R1/data/scene_datasets/mp3d'/house/(house+'.house'))
        objects = parse_house(path.read_text())
        rows, stats = propose(house,objects)
        reads.append({'house_id':house,'path':str(path.relative_to(ROOT)),'sha256':sha(path),
                      'access':'CPU_SEMANTIC_METADATA_TRAIN_ONLY'})
        coverage.append({'house_id':house,'object_count':len(objects),**stats})
        for rank,row in enumerate(rows):
            row['house_candidate_rank'] = rank
            row['split'] = 'FIT'
            row['source_house_sha256'] = reads[-1]['sha256']
            row['shared_split_sha256'] = sha(split_path)
            validate_quality(row)
            manifests.append(row)
        if not rows:
            ledger.append({'house_id':house,'stage':'cpu_semantic_planning','status':'metadata_ineligible',
                           'reason':'NO_DECLARED_FOUR_ROLE_COMBINATION','scientific_failure':False})
    # Round-robin houses permits a cross-house first batch, not 32 same-house jobs.
    manifests.sort(key=lambda r:(r['house_candidate_rank'],r['house_id']))
    for ordinal,row in enumerate(manifests):
        row['ordinal'] = ordinal
    summary = {'schema':'q35n.mechanism_scale_candidates.v1','source_sha256':sha(source),
        'shared_split_sha256':sha(split_path),'fit_houses_read':len(fit),
        'houses_with_candidates':sum(r['bounded_manifest_proposals']>0 for r in coverage),
        'semantic_candidates':len(manifests),'generated_physical_families':0,
        'mechanism_train_ready_families':0,'official_val_or_test_read':False,
        'heldout_house_assets_read':False,'runtime_executed':False,'scientific_pass':False}
    return {'summary':summary,'candidates':manifests,'coverage':coverage,'source_reads':reads,'failure_ledger':ledger}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--split',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(HERE) or output == HERE or output.exists():
        raise ValueError('FRESH_ISOLATED_OUTPUT_REQUIRED')
    result = build(args.split)
    output.mkdir(parents=True)
    for name,value in result.items():
        with (output/(name+'.json')).open('x') as f:
            json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(result['summary']))
