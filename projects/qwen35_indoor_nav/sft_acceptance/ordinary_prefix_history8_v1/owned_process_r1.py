"""Race-aware supervision of a directly spawned child, with PID-reuse-safe signaling."""
import os
import ctypes
import platform
from pathlib import Path
import signal
import subprocess

def pidfd_open(pid):
    if hasattr(os, 'pidfd_open'):
        return os.pidfd_open(pid)
    if platform.machine() != 'x86_64':
        raise RuntimeError('PIDFD_COMPAT_ARCH_UNSUPPORTED')
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    result = libc.syscall(434, ctypes.c_int(pid), ctypes.c_uint(0))
    if result < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return int(result)

def pidfd_send(fd, sig):
    if hasattr(signal, 'pidfd_send_signal'):
        return signal.pidfd_send_signal(fd, sig)
    if platform.machine() != 'x86_64':
        raise RuntimeError('PIDFD_COMPAT_ARCH_UNSUPPORTED')
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    result = libc.syscall(424, ctypes.c_int(fd), ctypes.c_int(sig), ctypes.c_void_p(), ctypes.c_uint(0))
    if result < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))

class IdentityConflict(RuntimeError):
    pass

def inspect(pid):
    path = Path('/proc') / str(pid)
    try:
        stat = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] in ('Z', 'X'):
            return {'pid': pid, 'start': int(stat[19]), 'state': stat[0]}
        argv = (path / 'cmdline').read_bytes().decode().split('\0')[:-1]
        cwd = str((path / 'cwd').resolve(strict=True))
        after = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        if after[0] in ('Z', 'X'):
            return {'pid': pid, 'start': int(after[19]), 'state': after[0]}
        if stat[19] != after[19]:
            raise IdentityConflict('PID_REUSED_DURING_INSPECTION')
        return {'pid': pid, 'start': int(stat[19]), 'state': after[0], 'argv': argv, 'cwd': cwd}
    except (FileNotFoundError, ProcessLookupError):
        return None

def stable(row):
    return {key: row[key] for key in ('pid', 'start', 'argv', 'cwd')}

class OwnedChild:
    def __init__(self, proc, expected_argv, expected_cwd):
        if not isinstance(proc, subprocess.Popen):
            raise TypeError('DIRECT_CHILD_HANDLE_REQUIRED')
        self.proc = proc
        self.fd = pidfd_open(proc.pid)
        self.signals = []
        try:
            row = inspect(proc.pid)
            if not row or row['state'] in ('Z', 'X'):
                raise IdentityConflict('CHILD_EXITED_BEFORE_IDENTITY_REGISTRATION')
            if row['argv'] != list(expected_argv) or row['cwd'] != str(expected_cwd):
                raise IdentityConflict('UNEXPECTED_CHILD_ENTRY')
            self.identity = stable(row)
        except BaseException:
            os.close(self.fd)
            self.fd = None
            raise

    def check(self, reader=inspect):
        if self.proc.poll() is not None:
            return 'exited'
        row = reader(self.proc.pid)
        if row is None or row['state'] in ('Z', 'X'):
            # Child can die between poll() and reading procfs. Reap via Popen;
            # do not infer an exit code from the vanished/empty procfs fields.
            try:
                self.proc.wait(timeout=.05)
                return 'exited'
            except subprocess.TimeoutExpired:
                return 'exiting'
        if stable(row) != self.identity:
            if self.proc.poll() is not None:
                return 'exited'
            again = reader(self.proc.pid)
            if again is None or again['state'] in ('Z', 'X'):
                return 'exiting'
            raise IdentityConflict('LIVE_CHILD_IDENTITY_CHANGED')
        return 'live'

    def send(self, sig):
        if self.check() != 'live':
            return False
        # pidfd targets the originally attached process, never a reused PID.
        try:
            pidfd_send(self.fd, sig)
        except ProcessLookupError:
            # Natural exit after the identity check is not a cleanup failure.
            self.proc.poll()
            return False
        self.signals.append(int(sig))
        return True

    def cleanup(self, timeout=5):
        error = None
        try:
            if self.check() == 'live':
                self.send(signal.SIGTERM)
            if self.proc.poll() is None:
                try:
                    self.proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    if self.check() == 'live':
                        self.send(signal.SIGKILL)
                        self.proc.wait(timeout=timeout)
        except BaseException as exc:
            error = repr(exc)
        finally:
            if self.fd is not None:
                os.close(self.fd)
                self.fd = None
        return dict(pid=self.proc.pid, exit_code=self.proc.poll(),
                    exited=self.proc.poll() is not None,
                    signals=self.signals, cleanup_error=error)

