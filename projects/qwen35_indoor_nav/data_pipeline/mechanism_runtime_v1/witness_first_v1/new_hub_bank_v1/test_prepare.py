"""CPU mocks and source differentials only; no fabricated scene/run files."""
import copy
import importlib.util
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('newhub_bank_test',HERE/'prepare.py');p=importlib.util.module_from_spec(s);s.loader.exec_module(p)
def fixture():
    positions=[[i*2.,0.,0.] for i in range(1,5)]
    cfg={'new_hub_plan':{'house_id':p.HOUSE,'limit':4,'selected_positions':positions,'excluded_positions':[[0.,0.,0.]],
         'scene_group_id':'mp3d:'+p.HOUSE,'house_split':'FIT'},
         'candidates':[{'house_id':p.HOUSE,'split':'FIT','source_positions':positions}]}
    result={'house_id':p.HOUSE,'status':'SCOUT_COMPONENT_BANK_COMPLETE','selected_hubs':4,'hubs':[{}, {}, {}, {}]}
    frozen={'hubs':[{'position':pos,'yaw_bin':0} for pos in positions],'saved_before_any_house_action':True,
        'geometry':{'all_four_runtime_geometry_retained':True,'frozen_selected_positions':positions}}
    return cfg,result,frozen
class Tests(unittest.TestCase):
    def test_exact_reverse(self):
        original=p.PREVIOUS.read_text();source=p.adapt(original)
        for a,b in reversed(p.patches()):source=source.replace(b,a)
        self.assertEqual(source,original)
    def test_changed_source_rejected(self):
        with self.assertRaises(ValueError):p.adapt(p.PREVIOUS.read_text()+'\n')
    def test_only_new_scout_source(self):
        m=p.load();self.assertEqual(m.SCOUT,p.SOURCE);self.assertEqual(m.HOUSES,[p.HOUSE]);self.assertEqual(m.HERE,HERE)
    def test_original_bank_algorithm_roundtrip(self):
        m=p.load();original=(m.OLD/'prepare.py').read_text();source=m.adapt(original)
        for a,b in reversed(m.patches()):source=source.replace(b,a)
        self.assertEqual(original,source)
    def test_four_mock_geometry_hubs_pass_structure_only(self):p.require_four_actual_hubs(*fixture())
    def test_fewer_than_four_refused(self):
        a,b,c=fixture();b['selected_hubs']=3
        with self.assertRaises(AssertionError):p.require_four_actual_hubs(a,b,c)
    def test_old_hub_overlap_refused(self):
        a,b,c=fixture();a['new_hub_plan']['excluded_positions']=[[2.,0.,0.]]
        with self.assertRaises(AssertionError):p.require_four_actual_hubs(a,b,c)
    def test_rotation_only_not_new_hub(self):
        a,b,c=fixture();c['hubs'][1]['position']=c['hubs'][0]['position'];c['hubs'][1]['yaw_bin']=1
        with self.assertRaises(AssertionError):p.require_four_actual_hubs(a,b,c)
    def test_group_change_refused(self):
        a,b,c=fixture();a['new_hub_plan']['house_split']='INTERNAL_DEV'
        with self.assertRaises(AssertionError):p.require_four_actual_hubs(a,b,c)
    def test_unknown_live_geometry_refused(self):
        a,b,c=fixture();c['geometry']['all_four_runtime_geometry_retained']=False
        with self.assertRaises(AssertionError):p.require_four_actual_hubs(a,b,c)
    def test_active_source_refused_before_output(self):
        m=p.load();saved=[];m.read=lambda path:{'status':'STARTED','error':None};m.save=lambda *args:saved.append(args)
        with self.assertRaises(AssertionError):m.main()
        self.assertEqual(saved,[])
    def test_missing_source_refused_before_output(self):
        m=p.load();saved=[]
        def missing(path):raise FileNotFoundError('CPU_MOCK_SOURCE_NOT_CLOSED')
        m.read=missing;m.save=lambda *args:saved.append(args)
        with self.assertRaises(FileNotFoundError):m.main()
        self.assertEqual(saved,[])
    def test_thresholds_retained(self):
        m=p.load();source=m.adapt((m.OLD/'prepare.py').read_text())
        for token in ('c.select_programs(viable,24)','maxcont>160','max(estimates.values())>160','c.solve(components','c.compiler.complete(trace)'):
            self.assertIn(token,source)

if __name__=='__main__':unittest.main()
