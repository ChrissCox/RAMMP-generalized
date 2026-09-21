"""Durable container exclusion beside the shared GPU-cache flock.

Callers must hold ``rammp-commissioning-planner.lock`` throughout these calls
and their container lifecycle. The flock handles concurrent live clients; this
journal survives client exit or SIGKILL while a Docker worker may still exist.
An earlier client's container is inspected only, never adopted or removed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import stat
import subprocess

from ..contracts import strict_loads
from .rolling import MotionError


JOURNAL_NAME = 'rammp-commissioning-planner-container.json'
_OWNED_NAME = re.compile(r'rammp-(?:warm-planner|reactive-probe)-[0-9a-f]{12}')


class GpuLeaseJournal:
    def __init__(self, cache):
        self.path = Path(cache)/JOURNAL_NAME

    @staticmethod
    def _name(value):
        if not isinstance(value, str) or _OWNED_NAME.fullmatch(value) is None:
            raise MotionError('GPU lease journal has an invalid owned container name')
        return value

    def _read(self):
        try:
            fd = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise MotionError('GPU lease journal cannot be read safely') from exc
        try:
            metadata = os.fstat(fd)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 4096:
                raise MotionError('GPU lease journal must be a bounded regular file')
            with os.fdopen(fd, 'rb', closefd=False) as source:
                value = strict_loads(source.read(4097), max_bytes=4096)
            if (not isinstance(value, dict) or set(value) != {'protocol', 'container_name'}
                    or type(value['protocol']) is not int or value['protocol'] != 1):
                raise MotionError('GPU lease journal is malformed')
            return self._name(value['container_name'])
        except (ValueError, RuntimeError) as exc:
            raise MotionError('GPU lease journal is malformed') from exc
        finally:
            os.close(fd)

    def _sync_directory(self):
        fd = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    @staticmethod
    def _require_absence(name):
        try:
            result = subprocess.run(['docker', 'ps', '-a', '--filter', 'name=^/'+name+'$',
                '--format', '{{.Names}}'], capture_output=True, text=True, timeout=15)
        except (OSError, subprocess.SubprocessError) as exc:
            raise MotionError('Prior GPU container absence is unproven; durable planner lease retained') from exc
        if result.returncode != 0 or not isinstance(result.stdout, str) or result.stdout.strip():
            raise MotionError('Prior GPU container absence is unproven; durable planner lease retained')

    def reconcile(self):
        """Refuse reuse after a crash until Docker proves the recorded name absent."""
        previous = self._read()
        if previous is not None:
            self._require_absence(previous)
            self.path.unlink()
            self._sync_directory()

    def reserve(self, name):
        """Persist before Popen/run; a malformed/partial journal fails closed."""
        self._name(name)
        try:
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        except OSError as exc:
            raise MotionError('GPU cache has an unresolved durable planner lease') from exc
        try:
            with os.fdopen(fd, 'w', closefd=False) as target:
                json.dump({'protocol': 1, 'container_name': name}, target)
                target.write('\n')
                target.flush()
                os.fsync(fd)
            self._sync_directory()
        finally:
            os.close(fd)

    def release(self, name):
        """Clear only our unchanged journal after a positive exact-name absence."""
        if self._read() != self._name(name):
            raise MotionError('GPU lease journal changed during the owned session')
        self._require_absence(name)
        self.path.unlink()
        self._sync_directory()
