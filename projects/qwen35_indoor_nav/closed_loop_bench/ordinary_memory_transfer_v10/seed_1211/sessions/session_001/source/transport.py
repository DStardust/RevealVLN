"""Idle simulator channels may outlive another arm's full trajectory and audit.

The parent retains its180-second request-response deadline and the launcher its
900-second progress watchdog. Waiting for a new request is not failed progress.
"""
import socket


def request_socket(fd):
    sock = socket.socket(fileno=fd)
    sock.settimeout(None)
    return sock


def close_stream(stream):
    try:
        stream.close()
    except (BrokenPipeError, ConnectionResetError):
        pass  # Preserve the recorded primary failure and finish owned cleanup.
