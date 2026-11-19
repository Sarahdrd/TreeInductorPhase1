"""Immutable checksummed attempts and one OS-locked writer per run directory."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import uuid

from .infrastructure import digest, json_bytes


class IntegrityError(ValueError):
    pass


def filesystem_path(path):
    """Extended Windows paths support deep OneDrive and sandbox directories."""
    path = Path(path).resolve()
    value = str(path)
    if os.name == 'nt' and not value.startswith('\\\\?\\'):
        value = '\\\\?\\UNC\\' + value[2:] if value.startswith('\\\\') else '\\\\?\\' + value
    return Path(value)


@contextmanager
def writer_lock(root):
    """Kernel releases the lock on process exit; never delete the lock inode."""
    root = filesystem_path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.writer.lock').open('a+b') as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError('Another writer owns this run; concurrent writers are prohibited') from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def atomic_write(path, payload):
    """Caller holds writer_lock. fsync contents then rename on the same volume.

    Existing final files are never replaced. Interrupted temporary files are
    ignored as uncommitted attempts; no successful result can reside only there.
    """
    path = filesystem_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    envelope = dict(payload=payload, sha256=digest(json_bytes(payload)))
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('xb') as stream:
        stream.write(json_bytes(envelope))
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        raise FileExistsError(path)
    os.rename(temporary, path)
    if os.name != 'nt':
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def read_verified(path):
    try:
        envelope = json.loads(Path(path).read_text(encoding='utf-8'))
        if set(envelope) != {'payload', 'sha256'}:
            raise ValueError('Invalid envelope')
        if digest(json_bytes(envelope['payload'])) != envelope['sha256']:
            raise ValueError('Checksum mismatch')
        return envelope['payload']
    except (ValueError, TypeError, KeyError, OSError) as exc:
        raise IntegrityError(f'Invalid checkpoint {path}: {exc}') from exc


class Store:
    def __init__(self, root, manifest):
        self.root, self.manifest = filesystem_path(root), manifest

    def check_manifest(self):
        path = self.root / 'run.json'
        if path.exists():
            if read_verified(path) != self.manifest:
                raise IntegrityError('Run provenance changed (code/config/environment/data); use a new experiment ID')
        elif (self.root / 'records').exists():
            raise IntegrityError('Records exist without a committed run manifest')

    def initialize(self):
        self.check_manifest()
        if not (self.root / 'run.json').exists():
            atomic_write(self.root / 'run.json', self.manifest)

    def folder(self, key):
        dataset, model, seed, repeat, fold = key
        return self.root / 'records' / dataset / model / f'seed_{seed}' / f'r{repeat}_f{fold}'

    def attempts(self, key):
        folder = self.folder(key)
        files = sorted(folder.glob('attempt_*.json'))
        records = []
        for index, path in enumerate(files, 1):
            if path.name != f'attempt_{index:06d}.json':
                raise IntegrityError(f'Non-contiguous or duplicate attempt sequence: {folder}')
            record = read_verified(path)
            if record.get('attempt') != index:
                raise IntegrityError(f'Attempt number mismatch: {path}')
            if records and records[-1]['execution_status'] == 'success':
                raise IntegrityError(f'Attempt follows a completed success: {folder}')
            records.append(record)
        unexpected = [p for p in folder.glob('*') if p.is_file() and p not in files and not p.name.endswith('.tmp')]
        if unexpected:
            raise IntegrityError(f'Unexpected record file: {unexpected[0]}')
        return records

    def save(self, key, record):
        previous = self.attempts(key)
        if previous and previous[-1]['execution_status'] == 'success':
            raise IntegrityError('Successful fold cannot be overwritten or repeated')
        if record['attempt'] != len(previous) + 1:
            raise IntegrityError('Attempt sequence changed')
        atomic_write(self.folder(key) / f"attempt_{record['attempt']:06d}.json", record)
