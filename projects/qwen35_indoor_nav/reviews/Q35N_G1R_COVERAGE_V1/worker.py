import hashlib
import importlib.util
import json
from pathlib import Path
import time

OUT=Path(__file__).resolve().parent
LINE=OUT.parents[1]
ORIGINAL=LINE/'reviews/Q35N_G1F_MINIMAL_FAMILY_REPLAY_ACCEPTANCE_V2'
source=(ORIGINAL/'engine.py').read_text()
assert source.count('cfg.gpu_device_id=2;')==1
source=source.replace('cfg.gpu_device_id=2;','cfg.gpu_device_id=3;')
s=importlib.util.spec_from_file_location('frozen_core',ORIGINAL/'engine.py')
core=importlib.util.module_from_spec(s)
exec(compile(source,str(ORIGINAL/'engine.py'),'exec'),core.__dict__)
core.OUT=OUT


class Engine(core.Engine):
    def discovery(self):
        pool=self.u_pool();attempts=0
        for tail in core.TAILS:
            for yaw in range(24):
                for index,u in enumerate(pool):
                    if attempts>=256:raise core.Reject('DISCOVERY_ATTEMPTS_EXHAUSTED',attempts=attempts)
                    attempts+=1;self.counts['family_candidates']=attempts
                    try:
                        if u['position'] is None:raise core.Reject('U_NONFINITE_SNAP')
                        base=self.candidate_base(u['position'],yaw)
                        candidate=self.complete_candidate(u,yaw,base,tail)
                        core.save('FROZEN_CANDIDATE.json',candidate)
                        return candidate
                    except core.Reject as e:
                        core.log('DISCOVERY_ATTEMPTS.jsonl',{'attempt':attempts,'u_index':index,'u':u,
                            'yaw_bin':yaw,'tail':tail,'status':'REJECTED','reason':e.code,'detail':e.detail})
                    if attempts%8==0:
                        core.save('PROGRESS.json',{'attempts':attempts,'last_u_index':index,'counts':dict(self.counts)})
                        print(f'{attempts}/256 completed',flush=True)


def main():
    assert not (OUT/'result.json').exists()
    core.save('SOURCE_DERIVATION.json',{'original':str(ORIGINAL/'engine.py'),'sha256':hashlib.sha256((ORIGINAL/'engine.py').read_bytes()).hexdigest(),
        'literal_patch':{'from':'cfg.gpu_device_id=2;','to':'cfg.gpu_device_id=3;'},'derived_source_sha256':hashlib.sha256(source.encode()).hexdigest(),
        'algorithm_override':'Engine.discovery changes enumeration only'})
    eng=None;started=time.time()
    try:
        eng=Engine('coverage');eng.preview()
        candidate=eng.discovery()
        core.save('result.json',{'decision':'FROZEN_CANDIDATE_READY_FOR_CERTIFICATION','candidate_hash':candidate['frozen_candidate_hash'],
            'family_certified':False,'scientific_pass':False})
    except core.Reject as e:
        core.save('result.json',{'decision':e.code,'detail':e.detail,'family_certified':False,'scientific_pass':False})
    finally:
        if eng:eng.close()
        core.save('WALL.json',{'seconds':time.time()-started})


if __name__=='__main__':main()
