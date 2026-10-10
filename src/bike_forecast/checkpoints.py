"""Atomic stage directories, hash verification and process exclusion."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import tempfile
from .data import sha256


class Paused(Exception):
    pass


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temporary, path)


def freeze(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('Frozen manifest mismatch: use a new output directory')
    else:
        atomic_json(path, value)


@contextmanager
def exclusive_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open('a+b')
    locked = False
    try:
        if os.name == 'nt':
            import msvcrt
            if path.stat().st_size == 0:
                handle.write(b'0'); handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        locked = True
        yield
    except (BlockingIOError, PermissionError) as exc:
        if not locked:
            raise RuntimeError('Another worker already holds the experiment lock') from exc
        raise
    finally:
        if locked:
            if os.name == 'nt':
                handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


class Stages:
    def __init__(self, root, max_stages=None):
        self.root = Path(root)
        self.max_stages = max_stages
        self.built = 0
        self.reused = 0

    def get(self, name, builder):
        path = self.root / name
        if path.exists():
            receipt = json.loads((path / 'complete.json').read_text())
            actual = {str(f.relative_to(path)): sha256(f) for f in sorted(path.rglob('*'))
                      if f.is_file() and f.name != 'complete.json'}
            if actual != receipt['files'] or receipt['stage'] != name:
                raise ValueError(f'Corrupt checkpoint: {name}')
            self.reused += 1
            return path
        if self.max_stages is not None and self.built >= self.max_stages:
            raise Paused(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Temporary work is never recognized as a completed stage.
        temporary = Path(tempfile.mkdtemp(prefix='.incomplete-', dir=path.parent))
        try:
            builder(temporary)
            files = {str(f.relative_to(temporary)): sha256(f) for f in sorted(temporary.rglob('*')) if f.is_file()}
            atomic_json(temporary / 'complete.json', {'stage': name, 'files': files})
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        self.built += 1
        print(f'Completed stage: {name}', flush=True)
        atomic_json(self.root / 'progress.json', {'last_completed': name,
                     'new_stages_this_run': self.built, 'reused_stages_this_run': self.reused})
        return path
