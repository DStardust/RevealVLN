"""Bounded G0R dependency/build commands; writes only inside Q35N."""
import datetime
import email
import email.policy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

OUT = Path(__file__).resolve().parent
LINE = OUT.parents[1]
ENV = LINE / '.envs/q35n_habitat_v017_g0r'
CACHE = LINE / '.cache/q35n_habitat_v017_g0r'
SRC = LINE / 'runtime/q35n_habitat_v017_g0r/src/habitat-sim'
WHEELS = CACHE / 'wheelhouse'
PY = str(ENV / 'bin/python3')
OLD = OUT.parent / 'Q35N_G0R_RUNTIME_SETUP_ACCEPTANCE_V1'

def save(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')

def inventory():
    records = []
    for path in sorted(WHEELS.glob('*.whl')):
        with zipfile.ZipFile(path) as z:
            meta = email.message_from_bytes(z.read(next(n for n in z.namelist() if n.endswith('.dist-info/METADATA'))), policy=email.policy.default)
        records.append({'file': path.name, 'bytes': path.stat().st_size,
                        'sha256': hashlib.file_digest(path.open('rb'), 'sha256').hexdigest() if sys.version_info >= (3,11) else hashlib.sha256(path.read_bytes()).hexdigest(),
                        'name': meta['Name'], 'version': meta['Version'], 'license': meta['License'],
                        'license_expression': meta['License-Expression'],
                        'requires_dist': meta.get_all('Requires-Dist', [])})
    save('WHEEL_INVENTORY.json', records)
    assert sum(r['bytes'] for r in records) < 2 * 1024**3
    return records

def run(step):
    assert LINE == Path('/mnt/data_nas/deeprobotics/daiyang/vla/projects/qwen35_indoor_nav')
    for path in (OUT, ENV, CACHE, SRC):
        assert path.resolve().is_relative_to(LINE)
    temp = CACHE / 'tmp_recovery'
    temp.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for key in ('PYTHONPATH','PYTHONHOME','PIP_TARGET','PIP_PREFIX','PIP_INDEX_URL','PIP_EXTRA_INDEX_URL','PIP_TRUSTED_HOST','CMAKE_PREFIX_PATH','CONDA_PREFIX','LD_LIBRARY_PATH'):
        env.pop(key, None)
    env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PIP_CONFIG_FILE='/dev/null',
               TMPDIR=str(temp), XDG_CACHE_HOME=str(CACHE), PIP_CACHE_DIR=str(CACHE / 'pip'),
               NUMBA_CACHE_DIR=str(CACHE / 'numba'), MPLCONFIGDIR=str(CACHE / 'matplotlib'),
               PATH=f'{ENV}/bin:/usr/bin:/bin', CMAKE_BUILD_PARALLEL_LEVEL='4',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    pip = [PY, '-I', '-m', 'pip', '--isolated']
    cwd = LINE
    if step == 'download':
        cmd = pip + ['download','--index-url','https://pypi.org/simple','--no-cache-dir',
                     '--only-binary=:all:','--retries','2','--timeout','30','--progress-bar','off',
                     '--dest',str(WHEELS),'-r',str(OLD / 'requirements.direct.txt')]
    elif step in ('install', 'install_local'):
        records = inventory()
        assert records
        cmd = pip + ['install','--no-index','--no-cache-dir','--find-links',str(WHEELS),
                     '-r',str(OLD / 'requirements.direct.txt')]
        if step == 'install_local':
            cmd += ['--prefix', str(ENV)]
    elif step == 'configure_relocated':
        env['CXXFLAGS'] = '-DVERSION_INFO=\\"0.1.7\\"'
        cmd = ['cmake','-S',str(SRC / 'src'),'-B',str(SRC / 'build'),'-GNinja',
               '-DCMAKE_BUILD_TYPE=RelWithDebInfo', '-DBUILD_PYTHON_BINDINGS=ON',
               f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={SRC}/build/lib.linux-x86_64-cpython-310/habitat_sim/_ext',
               f'-DPYTHON_EXECUTABLE={PY}', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
               '-DBUILD_GUI_VIEWERS=OFF','-DTARGET_HEADLESS=ON','-DBUILD_TEST=OFF',
               '-DBUILD_WITH_BULLET=OFF','-DBUILD_DATATOOL=OFF','-DBUILD_WITH_CUDA=OFF']
    elif step in ('build', 'build_relocated', 'package_existing'):
        env.update(PIP_NO_INDEX='1', PIP_NO_BUILD_ISOLATION='1')
        if step in ('build_relocated', 'package_existing'):
            cmake_cache = (SRC / 'build/CMakeCache.txt').read_text()
            compile_commands = (SRC / 'build/compile_commands.json').read_text()
            assert '/mnt/daiyang/' not in cmake_cache + compile_commands
            for key in ('PYTHON_LIBRARY:FILEPATH=', 'PYTHON_INCLUDE_DIRS:INTERNAL=', 'PYTHON_LIBRARIES:INTERNAL='):
                value = next(line.split('=',1)[1] for line in cmake_cache.splitlines() if line.startswith(key))
                assert Path(value).resolve().is_relative_to(ENV)
            save('RELOCATED_BUILD_PATH_CHECK.json', {'pass':True,'legacy_path_references':0,
                 'cache_sha256':hashlib.sha256(cmake_cache.encode()).hexdigest(),
                 'compile_commands_sha256':hashlib.sha256(compile_commands.encode()).hexdigest()})
        cmd = [PY, 'setup.py', '--headless', '--no-update-submodules', '--skip-install-magnum',
               'build_ext', '--parallel','4', 'bdist_wheel']
        if step == 'package_existing':
            cmd = [PY,'setup.py','--headless','--no-update-submodules','--skip-install-magnum',
                   'build_ext','--build-temp',str(SRC/'build'),'--parallel','4',
                   'bdist_wheel','--skip-build']
        cwd = SRC
    elif step == 'install_native':
        built = sorted((SRC / 'dist').glob('habitat_sim-0.1.7-*.whl'))
        magnum_setup = [SRC / 'build/deps/magnum-bindings/src/python/setup.py']
        assert magnum_setup[0].is_file()
        assert len(built) == len(magnum_setup) == 1, (built, magnum_setup)
        save('NATIVE_INSTALL_INPUTS.json', {
            'habitat_wheel':str(built[0]), 'habitat_wheel_sha256':hashlib.sha256(built[0].read_bytes()).hexdigest(),
            'magnum_generated_setup':str(magnum_setup[0]),
            'source_commit':'856d4b08c1a2632626bf0d205bf46471a99502b7',
            'external_downloads':False})
        cmd = pip + ['install','--no-index','--no-cache-dir','--no-build-isolation','--no-deps',
                     '--prefix',str(ENV),str(built[0]),str(magnum_setup[0].parent)]
    elif step == 'check':
        cmd = pip + ['check']
    elif step == 'check_native':
        cmd = pip + ['check']
    else:
        raise ValueError(step)
    resultpath = OUT / f'{step}.result.json'
    assert not resultpath.exists(), 'Use a new step/version; preserve completed attempts'
    started = time.time()
    save(f'{step}.command.json', {'step':step,'command':cmd,'cwd':str(cwd),
                                'started':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                'proxy_enabled': bool(env.get('https_proxy') or env.get('HTTPS_PROXY'))})
    print(f'Starting {step}; log: {OUT / (step + ".log")}', flush=True)
    with (OUT / f'{step}.log').open('x') as log:
        proc = subprocess.Popen(['/usr/bin/time','-v'] + cmd, cwd=cwd, env=env,
                                stdout=log, stderr=subprocess.STDOUT)
        try:
            rc = proc.wait(timeout=7200 if step.startswith('build') else 1800)
        except BaseException:
            proc.terminate()
            proc.wait(timeout=30)
            raise
    save(f'{step}.result.json', {'returncode':rc,'elapsed_seconds':time.time()-started})
    if step == 'download' and rc == 0:
        inventory()
    print(f'{step}: returncode={rc}, elapsed={time.time()-started:.1f}s', flush=True)
    sys.exit(rc)

if __name__ == '__main__':
    run(sys.argv[1])
