import importlib.util,runpy,unittest,math,ast
from pathlib import Path
HERE=Path(__file__).resolve().parent
route=runpy.run_path(str(HERE/'route.py'))
class RouteTests(unittest.TestCase):
    def advance(self,targets,cursor,pos):return route['advance'](targets,cursor,lambda goal:math.dist(pos,goal))
    def test_start_and_intermediate_not_goal_stop(self):
        targets=[[0,0,0],[1,0,0],[2,0,0]]
        x=self.advance(targets,0,[0,0,0])
        self.assertEqual(x['cursor_after'],1);self.assertEqual(x['selected_index'],1)
        self.assertFalse(route['stop_at'](x,2.))
    def test_near_goal_cannot_skip_unvisited_waypoint(self):
        x=self.advance([[0,0,0],[10,0,0],[1,0,0]],0,[1,0,0])
        self.assertEqual(x['cursor_after'],0)
        self.assertFalse(route['stop_at'](x,0.))
    def test_reach_boundary_and_repeated_waypoints(self):
        x=self.advance([[0,0,0],[0,0,0]],0,[.35,0,0])
        self.assertEqual(x['cursor_after'],2);self.assertTrue(route['stop_at'](x,.35))
        y=self.advance([[0,0,0]],0,[.350001,0,0])
        self.assertEqual(y['cursor_after'],0);self.assertFalse(route['stop_at'](y,.350001))
    def test_completed_route_agent_later_moves_away(self):
        x=self.advance([[0,0,0],[1,0,0]],2,[5,0,0])
        self.assertEqual(x['cursor_after'],2);self.assertEqual(x['selected_distance'],4)
        self.assertFalse(route['stop_at'](x,4.))
    def test_no_cursor_regression(self):
        x=self.advance([[0,0,0],[1,0,0],[2,0,0]],2,[0,0,0])
        self.assertEqual(x['cursor_after'],2)
    def test_episode_isolation(self):
        a=self.advance([[0,0,0]],0,[0,0,0]);b=self.advance([[10,0,0]],0,[0,0,0])
        self.assertEqual(a['cursor_after'],1);self.assertEqual(b['cursor_after'],0)
    def test_unreachable_is_unknown_not_stop(self):
        x=route['advance']([[0,0,0]],0,lambda target:math.inf)
        self.assertIsNone(x['selected_distance']);self.assertFalse(route['stop_at'](x,0))
    def test_invalid_distance(self):
        for d in (-1,math.nan):
            with self.assertRaises(AssertionError):route['advance']([[0,0,0]],0,lambda target:d)
    def test_no_future_target_calls(self):
        called=[]
        def distance(p):called.append(p);return 1.
        x=route['advance']([[0,0,0],[1,0,0]],0,distance)
        self.assertEqual(called,[[0,0,0]])
    def test_actual_worker_separates_route_labels_from_inputs(self):
        text=(HERE/'worker.py').read_text()
        self.assertIn("value=k.payload(e['instruction']['instruction_text'],recent,history[-8:])",text)
        self.assertIn("rt=route_info(cursor,before['position'],idx,j,1);cursor=rt['cursor_after']",text)
        self.assertIn("rt=route_info(cursor,pose(agent)['position'],idx,j,2);cursor=rt['cursor_after']",text)
        self.assertIn("agent.set_state(backup,reset_sensors=False)",text)
        self.assertIn("route_d,after_distance)",text)
        self.assertNotIn("if k.stop_at(d)",text)
        payload=ast.parse(text)
        calls=[x for x in ast.walk(payload) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='payload']
        self.assertEqual(len(calls),1);self.assertEqual(len(calls[0].args),3)
if __name__=='__main__':unittest.main()
