import ast
import errno
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import common as c
import run
import safe_size
import transport
class Tests(unittest.TestCase):
    def test_disjoint_namespace(self):
        for shard in (2,4):
            self.assertEqual(c.shard_root(shard),HERE.parent/'auto_generation_v3_production'/f'shard_{shard:04d}')
            self.assertFalse(c.shard_root(shard).is_relative_to(HERE))
    def test_source_subset(self):
        jobs=transport.rescue_jobs();self.assertEqual([len(jobs[s]) for s in (2,4)],[365,998])
        self.assertEqual(sum(len(j['instruction_alias_episodes']) for rows in jobs.values() for j in rows),4088)
    def test_exact_original_safe_size(self):
        self.assertEqual(safe_size.measure.__code__.co_filename,str(c.BASE/'recovery_v4/safe_size.py'))
        self.assertIn("value=safe_size.measure(c.shard_root(shard))",transport.run_source())
        self.assertIn("safe_size.measure(HERE)['apparent_bytes_conservative']<3*1024**3",transport.run_source())
        self.assertIn("<48*1024**3",transport.run_source())
    def test_nested_race_reproduced_and_sibling_scope_handles_atom(self):
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            runtime=Path(folder);root=runtime/'production/shard_0002';(root/'content').mkdir(parents=True)
            p=root/'content'/('a'*64+'.png.tmp');p.touch()
            def lstat(path):
                if Path(path)==p:raise FileNotFoundError(errno.ENOENT,'fixture rename',str(p))
                return os.lstat(path)
            with self.assertRaises(FileNotFoundError) as ctx:safe_size.measure(runtime,lstat=lstat)
            self.assertEqual(ctx.exception.filename,str(p))
            measured=safe_size.measure(root,lstat=lstat)
            self.assertGreaterEqual(measured['apparent_bytes_conservative'],1024**2)
            self.assertEqual(measured['tolerated_atomic_disappearances'],['content/'+p.name])
    def test_formal_missing_still_fails(self):
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            root=Path(folder);p=root/'JOBS.json';p.touch()
            def lstat(path):
                if Path(path)==p:raise FileNotFoundError(errno.ENOENT,'fixture formal',str(p))
                return os.lstat(path)
            with self.assertRaises(FileNotFoundError):safe_size.measure(root,lstat=lstat)
    def test_temporary_permission_and_io_still_fail(self):
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            root=Path(folder);(root/'content').mkdir();p=root/'content'/('a'*64+'.png.tmp');p.touch()
            for exception in (PermissionError,OSError):
                def lstat(path):
                    if Path(path)==p:raise exception(errno.EACCES,'fixture',str(p))
                    return os.lstat(path)
                with self.assertRaises(exception):safe_size.measure(root,lstat=lstat)
    def test_gpu_guard_unchanged(self):
        for memory,processes in ((500,{1:600}),(5000,{2:10}),(1800,{1:1500,2:10})):
            with self.assertRaises(AssertionError):run.contexts(dict(memory_mib=memory,processes=processes),2)
    def test_budget_and_dead_pane(self):
        code=transport.run_source();self.assertIn("['prior_wall_seconds']",code)
        self.assertIn("['prior_worker_seconds']",code);self.assertNotIn("'sleep 24000')",code)
        compile(transport.prepare_source(),'prepare','exec')
    def test_exception_handler_records_filename_and_traceback(self):
        module=ast.parse(transport.run_source())
        handler=next(n for n in ast.walk(module) if isinstance(n,ast.ExceptHandler) and any(isinstance(x,ast.Import) and x.names[0].name=='traceback' for x in n.body))
        caught=[]
        body=ast.unparse(ast.Module(body=handler.body,type_ignores=[]))
        program='try:\n    raise FileNotFoundError(2,"fixture","/fixture/formal.json")\nexcept BaseException as exc:\n'+''.join('    '+line+'\n' for line in body.splitlines())
        env=dict(c=mock.Mock(save=lambda p,v:caught.append((p,v))),out=Path('fixture'),started=run.time.monotonic(),time=run.time)
        exec(program,env)
        self.assertEqual(caught[0][1]['filename'],'/fixture/formal.json')
        self.assertIn('Traceback',caught[0][1]['traceback']);self.assertIn('FileNotFoundError',env['error'])
    def test_exact_approval_metadata_rejected(self):
        expected=dict(approved=True,gpu=6,shards=[2,4],input_lock_sha256='x',identity_sha256='y')
        with mock.patch.object(c,'immutable_verify',return_value={}),mock.patch.object(c,'approval_value',return_value=expected),mock.patch.object(c,'read',return_value=dict(expected,extra=True)):
            with self.assertRaisesRegex(AssertionError,'MAIN_APPROVAL_REQUIRED'):c.approved(6)
if __name__=='__main__':unittest.main(verbosity=2)
