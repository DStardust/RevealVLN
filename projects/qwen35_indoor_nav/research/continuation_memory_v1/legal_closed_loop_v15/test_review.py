"""CPU acceptance of whole-group resume, retained failures, and state seals."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parent))
import review as r


def main():
    protocol={'rollouts':[{'condition':i//18} for i in range(36)]}
    identity={'base_state_sha256':'base','head_states':{'head':'weights'}}
    def make_run(path,condition,complete=True,unchanged=True):
        path.mkdir()
        r.c.write(path/'METHOD_IDENTITY.json',identity)
        r.c.write(path/'STATE_SEAL.json',dict(identity,base_unchanged=unchanged,heads_unchanged=True,
            completed_ranks=list(range(condition*18,(condition+1)*18 if complete else condition*18+17))))
        if complete:r.c.write(path/f'CONDITION_{condition:03d}_AUDITS.json',[{'condition':condition}])
        return path
    def rejects(runs,reason):
        try:r.admitted_conditions(runs,protocol)
        except AssertionError as error:assert reason in str(error)
        else:raise AssertionError('INVALID_GROUP_ADMITTED')
    with tempfile.TemporaryDirectory(prefix='review_cpu_',dir=r.HERE) as tmp:
        root=Path(tmp)
        first=make_run(root/'first',0)
        partial=make_run(root/'partial',1,False)
        resumed=make_run(root/'resumed',1)
        admitted,_=r.admitted_conditions([first,partial,resumed],protocol)
        assert admitted=={0:first,1:resumed}
        rejects([first,make_run(root/'duplicate',0)],'DUPLICATE_COMPLETE_CONDITION')
        r.c.write(partial/'CONDITION_001_AUDITS.json',[{'condition':1}])
        rejects([first,partial],'UNSEALED_CONDITION')
        mutated=make_run(root/'mutated',1,unchanged=False)
        assert r.admitted_conditions([mutated],protocol)[0]=={}
        damaged=make_run(root/'badseal',1)
        seal=r.c.read(damaged/'STATE_SEAL.json');seal['head_states']={'head':'different'}
        r.c.write(damaged/'STATE_SEAL.json',seal)
        rejects([damaged],'HEAD_SEAL_MISMATCH')
        changed=make_run(root/'changed',1)
        new=deepcopy(identity);new['base_state_sha256']='newbase'
        r.c.write(changed/'METHOD_IDENTITY.json',new)
        seal=r.c.read(changed/'STATE_SEAL.json');seal['base_state_sha256']='newbase'
        r.c.write(changed/'STATE_SEAL.json',seal)
        try:r.admitted_conditions([first,changed],protocol)
        except AssertionError:pass
        else:raise AssertionError('DIFFERENT_BASES_MERGED')
    r.c.write(r.HERE/'REVIEW_CPU_TEST_RESULT.json',dict(passed=True,
        checks=['complete_groups_resume','partial_group_excluded_without_losing_complete_groups',
            'duplicate_group_rejected','unsealed_group_rejected','mutated_parameters_not_admitted',
            'seal_matches_initial_identity','different_bases_not_merged'],
        model_loaded=False,simulator_loaded=False))
    print('7 CPU resume/seal checks passed')


if __name__=='__main__':main()
