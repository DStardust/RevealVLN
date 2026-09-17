import copy
import json
from pathlib import Path
import tempfile
import sys
import unittest
from unittest import mock
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import common as c
import run
import safe_size
import merge


class RuntimeTests(unittest.TestCase):
    def test_manifest_partition_unchanged(self):
        plan=c.read(c.PARALLEL/'PLAN.json');self.assertEqual([r['route_count'] for r in plan['shards']],[250,244,253,253])
        jobs=[c.read(c.PARALLEL/r['jobs_path']) for r in plan['shards']]
        sets=[{j['physical_source_route_sha256'] for j in rows} for rows in jobs]
        self.assertEqual(len(set.union(*sets)),1000)
        self.assertEqual(sum(map(len,sets)),1000)
        houses=[{j['scene_id'] for j in rows} for rows in jobs]
        self.assertEqual(sum(map(len,houses)),len(set.union(*houses)))
        aliases=[(j['source'],str(e['episode_id'])) for rows in jobs for j in rows for e in j['instruction_alias_episodes']]
        self.assertEqual(len(aliases),len(set(aliases)))

    def test_exact_transport_rejects(self):
        for text in ('missing','oldold'):
            with self.assertRaises(AssertionError):c.exact(text,'old','new')

    def test_four_transports_compile_and_target_gpu(self):
        for s in range(4):
            text=c.worker_source(s);compile(text,'cpu_worker_transport','exec')
            self.assertIn(f'cfg.gpu_device_id={c.lane_for(s)};',text)
            self.assertIn('ASSET_HASH_CHANGED',text)
            self.assertIn(str(c.shard_root(s)),text)

    def test_safe_census_known_atoms(self):
        root=c.ROOT/'fixture'
        self.assertTrue(safe_size.known(root/'PROGRESS.json.pending',root))
        self.assertTrue(safe_size.known(root/'content'/('a'*64+'.png.tmp'),root))
        self.assertFalse(safe_size.known(root/'content'/'mystery.tmp',root))
        self.assertFalse(safe_size.known(root/'routes'/'x'/'result.json',root))

    def test_safe_census_errors_not_swallowed(self):
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            root=Path(folder)
            with mock.patch.object(safe_size.original.os,'scandir',side_effect=PermissionError('fixture')):
                with self.assertRaises(PermissionError):safe_size.measure(root)

    def test_identity_no_pidfd_requirement(self):
        self.assertNotIn('pidfd_open',(HERE/'run.py').read_text())
        fixture=dict(pid=1,starttime_ticks=10,proc_uid=0,cwd=str(c.ROOT),cmdline=['hold'],pane_id='%x',pane_target='target')
        actual={k:fixture[k] for k in ('pid','starttime_ticks','proc_uid','cwd','cmdline')}
        values={'#{pane_id}':'%x','#{pane_pid}':'1','#{pane_dead}':'0','#{pane_current_path}':str(c.ROOT)}
        with mock.patch.object(run,'process_identity',return_value=actual),mock.patch.object(run,'pane',side_effect=lambda i,k:values[k]):
            run.verify_identity(fixture)
            bad=dict(fixture,starttime_ticks=11)
            with self.assertRaises(AssertionError):run.verify_identity(bad)

    def test_xml_includes_graphics(self):
        xml='<nvidia_smi_log><gpu><uuid>u</uuid><fb_memory_usage><used>600 MiB</used></fb_memory_usage><utilization><gpu_util>0 %</gpu_util></utilization><processes><process_info><pid>1</pid><used_memory>246 MiB</used_memory><type>C</type></process_info><process_info><pid>2</pid><used_memory>300 MiB</used_memory><type>G</type></process_info></processes></gpu></nvidia_smi_log>'
        row=run.parse_gpu(xml,6,'u');self.assertEqual(row['processes'],{1:246,2:300})
        self.assertEqual(run.contexts(row,2),354)
        with self.assertRaises(AssertionError):run.contexts(dict(row,memory_mib=6000),2)

    def test_foreign_actual_task_rejected(self):
        with self.assertRaises(AssertionError):run.contexts(dict(memory_mib=1700,processes={1:1380,2:250}),2)

    def test_no_approval_no_output_or_gpu(self):
        with mock.patch.object(c,'approved',side_effect=AssertionError('approval')),mock.patch.object(Path,'mkdir') as mkdir,mock.patch.object(run,'gpu_snapshot') as gpu:
            with self.assertRaises(AssertionError):run.execute(6)
            mkdir.assert_not_called();gpu.assert_not_called()

    def test_closed_shard_not_replayed_in_worker(self):
        self.assertIn("assert not initial['complete']",(HERE/'worker.py').read_text())
        self.assertIn('UNFINISHED_ROUTE_REQUIRES_VERSIONED_RECOVERY',(HERE/'common.py').read_text())

    def test_alias_key_identity_source_scoped(self):
        j=dict(source='R2R',instruction_alias_episodes=[dict(episode_id=1),dict(episode_id=2)])
        self.assertEqual(merge.alias_keys(j),{('R2R','1'),('R2R','2')})

    def test_strict_auditor_unchanged(self):
        strict=c.load('test_original_auditor',c.BASE/'recovery_v1/audit.py')
        self.assertEqual(strict.strict.LEGAL,{'move_forward','turn_left','turn_right','STOP'})
        self.assertIn('else:assert delta<1e-6',(c.BASE/'audit.py').read_text())
        self.assertIn('except AssertionError:',(c.BASE/'recovery_v1/audit.py').read_text())

    def test_holder_exit_does_not_read_disappearing_cwd(self):
        identity=dict(pid=1,starttime_ticks=123)
        fields=['Z']+['0']*18+['123']
        with mock.patch.object(Path,'read_text',return_value='1 (holder) '+' '.join(fields)),mock.patch.object(run,'process_identity',side_effect=FileNotFoundError('cwd vanished')):
            run.wait_holder_exit(identity)

    def test_holder_exit_enoent_is_normal(self):
        with mock.patch.object(Path,'read_text',side_effect=FileNotFoundError()):
            self.assertFalse(run.same_process_running(1,123))

    def test_holder_pid_reuse_is_not_exit(self):
        fields=['S']+['0']*18+['124']
        with mock.patch.object(Path,'read_text',return_value='1 (holder) '+' '.join(fields)):
            with self.assertRaises(AssertionError):run.same_process_running(1,123)

    def test_pane_dead_transition_is_bounded_wait(self):
        values=iter(['0','0','1'])
        with mock.patch.object(run,'pane',side_effect=lambda i,k:'%x' if k=='#{pane_id}' else next(values)),mock.patch.object(run.time,'sleep') as sleep:
            run.wait_pane_dead({'pane_id':'%x'})
            self.assertEqual(sleep.call_count,2)

    def test_restore_failure_preserves_remain_on(self):
        identity=dict(pid=1,pane_id='%x',pane_target='target',cwd=str(c.ROOT),cmdline=['holder'],proc_uid=0)
        idle=dict(memory_mib=500,processes={1:246},utilization=0)
        with mock.patch.object(run,'drain_context',return_value=idle),mock.patch.object(run,'gpu_snapshot',return_value=idle),mock.patch.object(c,'save'),mock.patch.object(run,'wait_pane_dead'),mock.patch.object(run,'pane',side_effect=lambda i,k:'%x' if k=='#{pane_id}' else '2'),mock.patch.object(run,'call') as call,mock.patch.object(run.time,'sleep'),mock.patch.object(run,'process_identity',side_effect=FileNotFoundError('cwd missing')):
            result=run.restore_holder(identity,None,'off',HERE)
            self.assertFalse(result['restored'])
            self.assertFalse(any('set-option' in args.args for args in call.call_args_list))

    def test_restore_success_changes_remain_only_after_gpu_evidence(self):
        identity=dict(pid=1,pane_id='%x',pane_target='target',cwd=str(c.ROOT),cmdline=['holder'],proc_uid=0)
        idle=dict(memory_mib=500,processes={1:246},utilization=0)
        full=dict(memory_mib=29500,processes={1:246,2:29000},utilization=100)
        new=dict(pid=2,starttime_ticks=999,cmdline=['holder'],cwd=str(c.ROOT),proc_uid=0)
        with mock.patch.object(run,'drain_context',return_value=idle),mock.patch.object(run,'gpu_snapshot',side_effect=[idle,full]),mock.patch.object(c,'save'),mock.patch.object(run,'wait_pane_dead'),mock.patch.object(run,'pane',side_effect=lambda i,k:'%x' if k=='#{pane_id}' else '2'),mock.patch.object(run,'call') as call,mock.patch.object(run.time,'sleep'),mock.patch.object(run,'process_identity',return_value=new):
            result=run.restore_holder(identity,None,'off',HERE)
            self.assertTrue(result['restored'])
            self.assertIn('set-option',call.call_args_list[-1].args)

    def test_residual_holder_context_drains_before_guard(self):
        residual=dict(memory_mib=29500,processes={1:29000,2:246},utilization=0)
        idle=dict(memory_mib=500,processes={2:246},utilization=0)
        with tempfile.TemporaryDirectory(dir=HERE) as folder,mock.patch.object(run,'gpu_snapshot',side_effect=[residual,idle]),mock.patch.object(run.time,'sleep'):
            result=run.drain_context({},1,Path(folder),'TEST')
            self.assertEqual(result,idle)
            self.assertEqual(len((Path(folder)/'TEST_DRAIN.jsonl').read_text().splitlines()),2)

    def test_real_new_external_is_not_ignored_during_drain(self):
        residual=dict(memory_mib=31000,processes={1:29000,2:1380},utilization=0)
        with tempfile.TemporaryDirectory(dir=HERE) as folder,mock.patch.object(run,'gpu_snapshot',return_value=residual):
            with self.assertRaises(AssertionError):run.drain_context({},1,Path(folder),'TEST')

    def test_restore_never_uses_tmux_kill_pane(self):
        self.assertNotIn("'respawn-pane','-k'",(HERE/'run.py').read_text())
        self.assertIn('os.kill(pid,signal.SIGTERM)',(HERE/'run.py').read_text())

    def merge_fixture(self,root):
        for name in ('policy_1.json','policy_2.json','sup.json'):(root/name).touch()
        job=dict(job_id='j',source='R2R',source_sha256='a'*64,scene_id='house',physical_source_route_sha256='b'*64,
            instruction_alias_episodes=[dict(episode_id=1),dict(episode_id=2)])
        rows=[dict(policy_file=f'policy_{i}.json',supervision_file='sup.json',job_id='j',source='R2R',source_sha256='a'*64,
            scene_group='house',split='FIT',physical_source_route_sha256='b'*64,decisions=5) for i in (1,2)]
        return job,rows

    def test_merge_complete_alias_and_source_root(self):
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            root=Path(folder);job,rows=self.merge_fixture(root)
            result=merge.validate_accepted(root,job,rows,copy.deepcopy(rows),set(),set())
            self.assertEqual(len(result),2)
            self.assertTrue(all(r['sourceRoot']==str(root.relative_to(c.ROOT)) for r in result))
            with self.assertRaises(AssertionError):merge.validate_accepted(root,job,rows[:1],rows[:1],set(),set())

    def test_merge_duplicate_physical_or_alias_rejected(self):
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            root=Path(folder);job,rows=self.merge_fixture(root)
            with self.assertRaises(AssertionError):merge.validate_accepted(root,job,rows,rows,{'b'*64},set())
            with self.assertRaises(AssertionError):merge.validate_accepted(root,job,rows,rows,set(),{('R2R','1')})

    def test_merge_reaudit_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            root=Path(folder);job,rows=self.merge_fixture(root)
            wrong=copy.deepcopy(rows);wrong[0]['decisions']=6
            with self.assertRaises(AssertionError):merge.validate_accepted(root,job,rows,wrong,set(),set())


if __name__=='__main__':unittest.main()
