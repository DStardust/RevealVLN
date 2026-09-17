"""Explicitly permitted Habitat adapter. Importing this file is CPU-only.

Geometry proposals are NOT executed trajectories or visibility certificates.
Only step() moves the agent; reset/reconstruct are explicitly separate controls.
"""
import copy
import hashlib
import importlib
import math
from pathlib import Path
import re

PROJECT_ROOT = Path('/mnt/data_nas/deeprobotics/daiyang/vla')
NAMES = {'F': 'move_forward', 'L': 'turn_left', 'R': 'turn_right'}
RESERVED_MASKS = {0, 65535, 4294967295}


def validate_roles(roles):
    if not isinstance(roles, dict) or not roles:
        raise ValueError('ROLE_CONFIGURATION')
    result = copy.deepcopy(roles)
    for name, r in result.items():
        if (not isinstance(name, str) or not name or not isinstance(r, dict)
                or set(r) != {'mpcat40', 'room', 'raw_match'}
                or any(not isinstance(r[k], str) or not r[k] for k in ('mpcat40', 'room'))):
            raise ValueError('ROLE_CONFIGURATION')
        m = r['raw_match']
        if (not isinstance(m, dict) or set(m) != {'mode', 'value'}
                or m['mode'] not in ('exact', 'token')
                or not isinstance(m['value'], str) or not m['value']):
            raise ValueError('RAW_MATCH_CONFIGURATION')
        if m['value'] != normalize_raw(m['value']):
            raise ValueError('RAW_MATCH_NOT_NORMALIZED')
    return result


def normalize_raw(value):
    return ' '.join(value.lower().replace('#', ' ').split())


def role_matches(record, role):
    if record['mpcat40'] != role['mpcat40'] or record['room'] != role['room']:
        return False
    raw, match = normalize_raw(record['raw']), role['raw_match']
    if match['mode'] == 'exact':
        return raw == match['value']
    return re.search(r'\b' + re.escape(match['value']) + r'\b', raw) is not None


def validate_permit(scene_glb, gpu_device, permit):
    """Deny before *any* optional imports, scene read or GPU operation."""
    if not isinstance(permit, dict) or permit.get('runtime_allowed') is not True:
        raise PermissionError('RUNTIME_NOT_PERMITTED')
    if type(gpu_device) is not int or gpu_device < 0:
        raise ValueError('GPU_DEVICE')
    scene = Path(scene_glb)
    if not scene.is_absolute():
        raise PermissionError('SCENE_MUST_BE_ABSOLUTE')
    scene = scene.resolve(strict=True)
    if not scene.is_relative_to(PROJECT_ROOT) or not scene.is_file() or scene.suffix != '.glb':
        raise PermissionError('SCENE_OUT_OF_SCOPE')
    if permit.get('scene_glb') != str(scene) or permit.get('gpu_device') != gpu_device:
        raise PermissionError('PERMIT_ASSET_OR_GPU_MISMATCH')
    return scene


def turns(delta):
    delta %= 24
    return ['L'] * delta if delta <= 12 else ['R'] * (24-delta)


def heading(dx, dz):
    return int(round(math.atan2(-dx, -dz) * 12/math.pi)) % 24


def geometric_actions(points, yaw, center, max_actions=504):
    """Fixed 0.25m/15deg ideal geometry; no simulator, oracle visibility or moves.

    Each path waypoint is chased within 0.1875m horizontally. Vertical travel
    is not modelled: actual navmesh stepping/replay must validate all proposals.
    """
    if type(yaw) is not int or not 0 <= yaw < 24:
        raise ValueError('YAW_BIN')
    if len(points) < 1 or any(len(p) != 3 or not all(math.isfinite(float(x)) for x in p) for p in points):
        raise ValueError('PATH_POINTS')
    if len(center) != 3 or not all(math.isfinite(float(x)) for x in center):
        raise ValueError('TARGET_CENTER')
    x, _, z = map(float, points[0])
    result = []
    for waypoint in points[1:]:
        # Bound both action length and iteration count against degenerate paths.
        attempts = 0
        while math.hypot(float(waypoint[0])-x, float(waypoint[2])-z) > .1875:
            attempts += 1
            if attempts > max_actions:
                raise ValueError('GEOMETRIC_PATH_DID_NOT_CONVERGE')
            new_yaw = heading(float(waypoint[0])-x, float(waypoint[2])-z)
            result += turns(new_yaw-yaw) + ['F']
            yaw = new_yaw
            x -= .25*math.sin(yaw*math.pi/12)
            z -= .25*math.cos(yaw*math.pi/12)
            if len(result) > max_actions:
                raise ValueError('GEOMETRIC_ACTION_CAP')
    desired = heading(float(center[0])-x, float(center[2])-z)
    result += turns(desired-yaw) + ['L', 'R']
    if len(result) > max_actions:
        raise ValueError('GEOMETRIC_ACTION_CAP')
    return result


class HabitatBackend:
    def __init__(self, scene_glb, gpu_device, roles, content_store, permit):
        self.scene_glb = validate_permit(scene_glb, gpu_device, permit)
        self.roles = validate_roles(roles)
        if not callable(getattr(content_store, 'put_array', None)):
            raise ValueError('CONTENT_STORE_INTERFACE')
        self.content_store = content_store
        self.compiler_roles = {k: (r['mpcat40'], r['room']) for k, r in self.roles.items()}
        # No real imports occur before permission and configuration validation.
        self.hs = importlib.import_module('habitat_sim')
        self.np = importlib.import_module('numpy')
        self.quaternion = importlib.import_module('quaternion')
        hs = self.hs
        cfg = hs.SimulatorConfiguration()
        cfg.scene_id = str(self.scene_glb)
        cfg.gpu_device_id = gpu_device
        cfg.enable_physics = False
        cfg.allow_sliding = False
        ac = hs.agent.AgentConfiguration()
        ac.height, ac.radius = 1.5, .1
        sensors = []
        for name, kind in [('rgb', hs.SensorType.COLOR), ('semantic', hs.SensorType.SEMANTIC)]:
            s = hs.SensorSpec()
            s.uuid, s.sensor_type = name, kind
            s.resolution, s.position, s.orientation = [224, 224], [0, 1.25, 0], [0, 0, 0]
            s.parameters['hfov'] = '90'
            s.gpu2gpu_transfer = False
            sensors.append(s)
        ac.sensor_specifications = sensors
        ac.action_space = {name: hs.agent.ActionSpec(name, hs.agent.ActuationSpec(amount=.25 if key == 'F' else 15))
                           for key, name in NAMES.items()}
        self.sim = hs.Simulator(hs.Configuration(cfg, [ac]))
        self.counts = {'simulator_constructions': 1, 'primitive_actions': 0,
                       'trace_initializations': 0, 'explicit_resets': 0,
                       'observation_records': 0, 'explicit_reconstructions': 0}
        self._obs, self._closed = None, False
        self.route_diagnostics = []
        try:
            self.objects, self.eligible = self._inventory()
        except BaseException:
            self.close()
            raise

    def _inventory(self):
        objects, eligible = {}, {key: [] for key in self.roles}
        for obj in self.sim.semantic_scene.objects:
            if obj is None:
                continue
            idx = int(obj.id.rsplit('_', 1)[1])
            if idx in objects:
                raise ValueError('DUPLICATE_SEMANTIC_MASK_ID')
            region = obj.region
            record = {'id': obj.id, 'mask_id': idx, 'raw': obj.category.name('raw'),
                      'mpcat40': obj.category.name('mpcat40'),
                      'room': region.category.name() if region else None,
                      'region_id': region.id if region else None,
                      'center': obj.obb.center.tolist(), 'sizes': obj.obb.sizes.tolist()}
            if (not isinstance(record['raw'], str) or len(record['center']) != 3
                    or not all(math.isfinite(x) for x in record['center'])):
                raise ValueError('SEMANTIC_RECORD_INVALID')
            objects[idx] = record
            for key, role in self.roles.items():
                if role_matches(record, role):
                    if idx <= 0 or idx in RESERVED_MASKS:
                        raise ValueError('RESERVED_ELIGIBLE_MASK')
                    eligible[key].append(idx)
        return objects, {k: sorted(v) for k, v in eligible.items()}

    def _active(self):
        if self._closed:
            raise RuntimeError('BACKEND_CLOSED')

    def reset(self, position, yaw, seed):
        self._active()
        if (len(position) != 3 or not all(math.isfinite(float(x)) for x in position)
                or type(yaw) is not int or not 0 <= yaw < 24
                or type(seed) is not int or seed < 0):
            raise ValueError('RESET_ARGUMENTS')
        self.sim.seed(seed)
        state = self.hs.AgentState()
        state.position = self.np.asarray(position, dtype=self.np.float32)
        angle = yaw*math.pi/24
        state.rotation = self.np.quaternion(math.cos(angle), 0, math.sin(angle), 0)
        self.sim.initialize_agent(0, state)
        self.counts['trace_initializations'] += 1
        self._obs = self.sim.reset()
        self.counts['explicit_resets'] += 1

    def _pose(self):
        state = self.sim.get_agent(0).get_state()
        def base(s):
            return {'position': s.position.tolist(),
                    'rotation': self.quaternion.as_float_array(s.rotation).tolist()}
        result = base(state)
        result['sensors'] = {k: base(v) for k, v in state.sensor_states.items()}
        if set(result['sensors']) != {'rgb', 'semantic'}:
            raise ValueError('SENSOR_POSE_SET')
        return result

    def observe(self):
        self._active()
        if self._obs is None:
            raise RuntimeError('RESET_REQUIRED')
        np = self.np
        rgb = np.ascontiguousarray(self._obs['rgb'][:, :, :3])
        semantic = np.ascontiguousarray(self._obs['semantic'])
        if rgb.shape != (224, 224, 3) or rgb.dtype != np.uint8:
            raise ValueError('RGB_LAYOUT')
        if semantic.shape != (224, 224) or semantic.dtype != np.uint32:
            raise ValueError('SEMANTIC_LAYOUT')
        hashes = {}
        for arr, kind in [(rgb, 'rgb'), (semantic, 'semantic')]:
            expected = hashlib.sha256(arr.tobytes()).hexdigest()
            stored = self.content_store.put_array(arr, kind)
            returned = stored.get('pixel_sha256') if isinstance(stored, dict) else stored
            if returned != expected:
                raise ValueError('CONTENT_STORE_PIXEL_HASH_MISMATCH')
            hashes[kind+'_hash'] = expected
        ids, nums = np.unique(semantic, return_counts=True)
        pixels = {str(int(k)): int(v) for k, v in zip(ids, nums)}
        unknown = sorted(set(map(int, pixels))-set(self.objects)-RESERVED_MASKS)
        self.counts['observation_records'] += 1
        return dict(hashes, pixels=pixels, pose=self._pose(),
                    evidence_complete=not unknown, unknown_mask_ids=unknown)

    def step(self, action):
        self._active()
        if self._obs is None:
            raise RuntimeError('RESET_REQUIRED')
        if action not in NAMES:
            raise ValueError('PRIMITIVE_ACTION_ONLY')
        self._obs = self.sim.step(NAMES[action])
        self.counts['primitive_actions'] += 1
        # A missing collision field must not silently become a clean trajectory.
        if 'collided' not in self._obs:
            raise ValueError('COLLISION_SIGNAL_MISSING')
        return bool(self._obs['collided'])

    def reconstruct(self, pose):
        """Explicit numerical join; caller must certify pre/post <=1e-5 bounds.

        Uses the accepted engine's sensor inference path, not manual sensor edits.
        TraceRunner verifies that resulting full agent/sensor pose equals target.
        """
        self._active()
        if self._obs is None:
            raise RuntimeError('RESET_REQUIRED')
        if (set(pose) != {'position', 'rotation', 'sensors'} or len(pose['position']) != 3
                or len(pose['rotation']) != 4 or set(pose['sensors']) != {'rgb', 'semantic'}
                or not all(math.isfinite(float(x)) for x in pose['position']+pose['rotation'])
                or abs(sum(float(x)**2 for x in pose['rotation'])-1) > 1e-4):
            raise ValueError('RECONSTRUCTION_POSE')
        state = self.hs.AgentState()
        state.position = self.np.asarray(pose['position'], dtype=self.np.float32)
        state.rotation = self.np.quaternion(*pose['rotation'])
        self.sim.get_agent(0).set_state(state, reset_sensors=True, infer_sensor_states=True)
        self._obs = self.sim.get_sensor_observations()
        self.counts['explicit_reconstructions'] += 1

    def routes(self, position, yaw, role):
        """Deterministic geometric proposals. NO reset/step/follower/render calls."""
        self._active()
        if role not in self.eligible:
            raise ValueError('UNKNOWN_ROLE')
        np, hs = self.np, self.hs
        seen = set()
        for instance in sorted(self.eligible[role]):
            center = self.objects[instance]['center']
            for radius in (.75, 1.25):
                for angle in range(8):
                    original = [center[0]+radius*math.cos(angle*math.pi/4), center[1],
                                center[2]+radius*math.sin(angle*math.pi/4)]
                    target = self.sim.pathfinder.snap_point(np.asarray(original, dtype=np.float32))
                    record = {'role': role, 'instance': instance, 'radius': radius,
                              'angle': angle, 'original': original}
                    if not np.isfinite(target).all():
                        self.route_diagnostics.append(dict(record, status='NONFINITE_SNAP'))
                        continue
                    path = hs.ShortestPath()
                    path.requested_start = np.asarray(position, dtype=np.float32)
                    path.requested_end = target
                    if not self.sim.pathfinder.find_path(path):
                        self.route_diagnostics.append(dict(record, status='NO_NAVMESH_PATH'))
                        continue
                    try:
                        actions = geometric_actions([p.tolist() for p in path.points], yaw, center)
                    except ValueError as exc:
                        self.route_diagnostics.append(dict(record, status=str(exc)))
                        continue
                    key = tuple(actions)
                    if key in seen:
                        self.route_diagnostics.append(dict(record, status='DUPLICATE_ACTION_PROPOSAL'))
                        continue
                    seen.add(key)
                    self.route_diagnostics.append(dict(record, status='GEOMETRY_PROPOSAL_ONLY',
                                                       actions=len(actions), snapped=target.tolist()))
                    yield actions

    def snap_position(self, position):
        """Pure navmesh metadata; returns None on nonfinite snap, never moves."""
        self._active()
        if len(position) != 3 or not all(math.isfinite(float(x)) for x in position):
            raise ValueError('SNAP_POSITION')
        snapped = self.sim.pathfinder.snap_point(self.np.asarray(position, dtype=self.np.float32))
        return snapped.tolist() if self.np.isfinite(snapped).all() else None

    def close(self):
        if not getattr(self, '_closed', False):
            self.sim.close()
            self._closed = True
            self._obs = None


Backend = HabitatBackend
