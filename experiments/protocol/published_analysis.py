"""Read the explicitly curated review subset without raw evaluation stores.

Original full-analysis checksum manifests and immutable analysis modules are
unchanged. This display-only loader verifies every distributed artifact and
returns only the CSV tables listed in the separate publication manifest.
"""
import json
from pathlib import Path
import pandas as pd
from .infrastructure import digest
from .checkpoints import IntegrityError


def load_outputs(output):
    output = Path(output)
    manifest = json.loads((output/'published_checksums.json').read_text())
    tables = {}
    for name, expected in manifest['files'].items():
        path = output/name
        if digest(path.read_bytes()) != expected:
            raise IntegrityError('Published artifact changed: '+str(path))
        if path.suffix == '.csv':
            tables[path.stem] = pd.read_csv(path, float_precision='round_trip')
    return tables
