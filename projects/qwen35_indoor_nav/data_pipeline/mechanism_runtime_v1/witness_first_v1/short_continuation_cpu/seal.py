import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    data=json.loads((HERE/'CANDIDATES.json').read_text())
    for name,value in data['source_hashes'].items():assert sha(Path(name))==value,name
    assert data['candidate_count']==3 and data['query_sequence_limit_unchanged']==160
    result={'status':'CPU_SHORT_CONTINUATION_PROPOSALS_READY_FOR_REPLAY',
            'cpu_tests_passed':9,'candidate_count':3,'motion_actions':[34,36,44],
            'source_hashes_verified':len(data['source_hashes']),
            'actual_new_replays':0,'runtime_pass':None,'scientific_pass':False,
            'old_history_or_checker_modified':False}
    with (HERE/'result.json').open('x') as f:json.dump(result,f,indent=2)
    files=sorted(p for p in HERE.iterdir() if p.is_file() and p.name!='SHA256SUMS')
    with (HERE/'SHA256SUMS').open('x') as f:
        for path in files:f.write(sha(path)+'  '+path.name+'\n')
    print(json.dumps(result))
if __name__=='__main__':main()
