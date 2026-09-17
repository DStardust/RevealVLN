import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import json_stream
import envdrop_manifest as m
class Tests(unittest.TestCase):
    def test_stream_metadata_and_entries(self):
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            p=Path(d)/'test.gz'
            with gzip.open(p,'wt') as f:json.dump(dict(episodes=[dict(a=1),dict(b=2)],instruction_vocab={'x':1}),f)
            self.assertEqual(list(json_stream.episodes(p)),[dict(a=1),dict(b=2)])
    def test_split_and_count_selection(self):
        jobs=[dict(scene_id=h,physical_source_route_sha256=str(i)) for h in ('a','b') for i in range(10)]
        chosen=m.select(jobs,10);self.assertEqual(len(chosen),10);self.assertEqual([j['scene_id'] for j in chosen[:4]],['a','b','a','b'])
    def test_geometry_ignores_radius_and_quaternion_sign(self):
        e=dict(scene_id='s',start_position=[0,0,0],start_rotation=[0,0,0,1],reference_path=[[0,0,0],[1,0,0]],goals=[dict(position=[1,0,0],radius=3)])
        v=json.loads(json.dumps(e));v['start_rotation']=[0,0,0,-1];v['goals'][0]['radius']=2
        self.assertEqual(m.geometry_key(e),m.geometry_key(v))
    def test_source_pose_mismatch_rejected(self):
        e=dict(scene_id='s',instruction={'instruction_text':'go'},start_position=[1,0,0],start_rotation=[0,0,0,1],reference_path=[[0,0,0],[1,0,0]],goals=[dict(position=[1,0,0])])
        with self.assertRaises(AssertionError):m.validate(e)
if __name__=='__main__':unittest.main(verbosity=2)
