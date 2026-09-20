"""Atomic, durable, no-replace publication for a validated FAMILY payload."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Callable


class PublicationDurabilityUnconfirmed(RuntimeError):
    """The final name is visible but directory durability was not confirmed."""


def publish_bytes_no_replace(
    destination: Path,
    payload: bytes,
    *,
    fsync: Callable[[int], None] = os.fsync,
    link: Callable[..., None] = os.link,
) -> None:
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    destination = Path(destination)
    parent = destination.parent
    if not parent.is_dir():
        raise FileNotFoundError(f"parent directory is missing: {parent}")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(str(destination))

    fd, temporary_name = tempfile.mkstemp(prefix=".pending_", suffix=".json.tmp", dir=parent)
    temporary = Path(temporary_name)
    linked = False
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            fsync(stream.fileno())
        link(temporary, destination, follow_symlinks=False)
        linked = True
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_fd = os.open(parent, flags)
        try:
            fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException as original_error:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        except OSError as cleanup_error:
            raise RuntimeError(
                f"PUBLICATION_AND_CLEANUP_ERROR: original={original_error!r}; "
                f"cleanup={cleanup_error!r}; final_name_linked={linked}"
            ) from original_error
        if linked and isinstance(original_error, OSError):
            raise PublicationDurabilityUnconfirmed(str(destination)) from original_error
        raise
    else:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        except OSError as cleanup_error:
            raise RuntimeError(
                f"PUBLICATION_COMPLETE_BUT_TEMP_CLEANUP_FAILED:{destination}"
            ) from cleanup_error

