"""Pure CPU tests: handler methods invoked in memory; no listening service."""
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('training_monitor', HERE / 'server.py')
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='cpu_monitor_', dir=HERE)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / 'run_0001'
        self.run.mkdir()

    def write(self, path, value):
        path.write_text(json.dumps(value))

    def test_nested_metrics_missing_result_and_staleness(self):
        progress = self.run / 'PROGRESS.json'
        self.write(progress, {'status': 'training', 'unix': 5, 'cursor': {'steps': 10},
            'metrics': {'loss': 1.2, 'STOP': {'tp': 2, 'fn': 1}},
            'budget': {'max_steps': 100}, 'throughput': {'decisions_per_second': 3}})
        os.utime(progress, (100, 100))
        report = server.collect_status(self.root, now=300, include_gpu=False)
        self.assertEqual(report['status']['error'], 'MISSING')
        run = report['runs'][0]
        self.assertTrue(run['progress']['stale'])
        self.assertEqual(run['progress']['age_seconds'], 200)
        self.assertEqual(run['progress']['data']['metrics']['STOP']['tp'], 2)
        self.assertEqual(run['result']['error'], 'MISSING')

    def test_bad_json_is_explicit_and_no_exception_text_leaks(self):
        (self.run / 'PROGRESS.json').write_text('{broken super_secret_token')
        report = server.collect_status(self.root, include_gpu=False)
        self.assertFalse(report['runs'][0]['progress']['readable'])
        self.assertEqual(report['runs'][0]['progress']['error'], 'JSONDecodeError')
        self.assertNotIn('super_secret_token', json.dumps(report))

    def test_filter_credentials_and_nonfinite_values(self):
        self.write(self.root / 'STATUS.json', {'status': 'running', 'api_key': 'secret_A',
            'environment': {'PATH': 'secret_B'}, 'metrics': {'loss': float('nan'),
            'access_token': 'secret_C', 'forward_tokens': 123}, 'unknown_private': 'secret_D'})
        result = server.collect_status(self.root, include_gpu=False)
        encoded = json.dumps(result, allow_nan=False)
        self.assertNotIn('secret_', encoded)
        self.assertEqual(result['status']['data']['metrics']['forward_tokens'], 123)

    def test_checkpoint_only_name_and_known_shallow_directory(self):
        self.write(self.run / 'PROGRESS.json', {'checkpoint': '/private/path/step_100.pt'})
        result = server.collect_status(self.root, include_gpu=False)
        self.assertEqual(result['runs'][0]['checkpoint_name'], 'step_100.pt')
        (self.run / 'PROGRESS.json').unlink()
        folder = self.run / 'checkpoints'
        folder.mkdir()
        (folder / 'step_101.pt').write_bytes(b'never deserialize')
        self.assertEqual(server.collect_status(self.root, include_gpu=False)['runs'][0]['checkpoint_name'], 'step_101.pt')

    def test_symlink_escape_and_run_symlink_ignored(self):
        outside = self.root / 'elsewhere'
        outside.mkdir()
        (outside / 'STATUS.json').write_text('{"status":"not allowed"}')
        (self.root / 'STATUS.json').symlink_to(HERE / 'server.py')
        (self.root / 'run_escape').symlink_to(HERE)
        result = server.collect_status(self.root, include_gpu=False)
        self.assertEqual(result['status']['error'], 'ValueError')
        self.assertEqual([r['name'] for r in result['runs']], ['run_0001'])

    def test_oversize_file_is_rejected(self):
        self.write(self.root / 'PROBE.json', {'status': 'a' * 100})
        with mock.patch.object(server, 'MAX_JSON_BYTES', 20):
            probe = server.collect_status(self.root, include_gpu=False)['probe']
        self.assertEqual(probe['error'], 'ValueError')

    def test_gpu_timeout_is_two_seconds_and_unavailable_explicit(self):
        with mock.patch.object(server.subprocess, 'run', side_effect=subprocess.TimeoutExpired('nvidia-smi', 2)) as call:
            result = server.gpu_status()
        self.assertFalse(result['available'])
        self.assertEqual(result['error'], 'TimeoutExpired')
        self.assertEqual(call.call_args.kwargs['timeout'], 2)
        self.assertEqual(call.call_args.args[0][0], 'nvidia-smi')

    def test_gpu_output_is_numeric_only(self):
        fake = types.SimpleNamespace(returncode=0, stdout='0, 123, 24000, 50\nmalicious command\n')
        with mock.patch.object(server.subprocess, 'run', return_value=fake):
            result = server.gpu_status()
        self.assertEqual(result['devices'], [{'index': 0, 'used_mib': 123,
                                             'total_mib': 24000, 'utilization_percent': 50}])

    def handler(self, path, command='GET'):
        handler = object.__new__(server.make_handler(self.root))
        handler.path = path
        handler.command = command
        handler.respond = mock.Mock()
        return handler

    def test_fixed_routes_no_browsing_or_query_parameters(self):
        for path in ('/etc/passwd', '/../STATUS.json', '/%2e%2e/STATUS.json', '/api/status?file=STATUS.json'):
            h = self.handler(path)
            h.do_GET()
            self.assertEqual(h.respond.call_args.args[0], 404)
        for path in ('/', '/healthz', '/api/status'):
            h = self.handler(path)
            with mock.patch.object(server, 'gpu_status', return_value={'available': False}):
                h.do_GET()
            self.assertEqual(h.respond.call_args.args[0], 200)

    def test_write_methods_rejected(self):
        for method in ('do_POST', 'do_PUT', 'do_DELETE', 'do_PATCH'):
            h = self.handler('/api/status')
            getattr(h, method)()
            self.assertEqual(h.respond.call_args.args[0], 405)

    def test_head_returns_no_body_and_security_headers(self):
        h = object.__new__(server.make_handler(self.root))
        h.command, h.wfile = 'HEAD', io.BytesIO()
        h.send_response, h.send_header, h.end_headers = mock.Mock(), mock.Mock(), mock.Mock()
        h.respond(200, b'body')
        self.assertEqual(h.wfile.getvalue(), b'')
        self.assertIn(mock.call('Cache-Control', 'no-store'), h.send_header.call_args_list)
        self.assertEqual((server.HOST, server.PORT), ('127.0.0.1', 18766))


if __name__ == '__main__':
    unittest.main(verbosity=2)
