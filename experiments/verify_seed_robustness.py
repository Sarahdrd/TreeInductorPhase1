"""Run the complete notebook in an isolated output directory.

Preserve existing canonical files and notebook outputs; publish only the seed
robustness outputs after successful execution and predictive-result comparison.
"""
import hashlib
import copy
import os
from pathlib import Path
import shutil
import sys
import tempfile

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpec
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
root = Path(__file__).resolve().parents[1]
path = root / 'experiments' / 'sensitivity_ranking_test.ipynb'
output = root / 'results' / 'dataset_statistics'
original = nbformat.read(path, as_version=4)
canonical_hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in output.iterdir() if p.is_file()}
reference = pd.read_csv(output / 'dataset_experimental_profile.csv').set_index('dataset').sort_index()
os.environ['MPLBACKEND'] = 'module://matplotlib_inline.backend_inline'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

with tempfile.TemporaryDirectory(prefix='verify-seed-robustness-') as directory:
    temporary = Path(directory)
    notebook = copy.deepcopy(original)
    for cell in notebook.cells:
        if cell.cell_type != 'code':
            continue
        redirected_path = 'Path('+repr(str(temporary))+')'
        cell.source = cell.source.replace("PROJECT_ROOT / 'results' / 'dataset_statistics'", redirected_path)
        cell.source = cell.source.replace('PROJECT_ROOT / "results" / "dataset_statistics"', redirected_path)
        compile(cell.source, 'notebook cell', 'exec')
    manager = KernelManager()
    manager._kernel_spec = KernelSpec(argv=[sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
                                     display_name='Python', language='python')
    def progress(cell, cell_index, **kwargs):
        print(f'Executing cell {cell_index}: {cell.source[:85]!r}', flush=True)
    client = NotebookClient(notebook, km=manager, timeout=1800,
                            resources={'metadata': {'path': str(root)}}, on_cell_start=progress)
    client.execute()
    rerun = pd.read_csv(temporary / 'dataset_experimental_profile.csv').set_index('dataset').sort_index()
    for column in ['baseline_macro_f1_mean', 'cv_variability', 'predictive_error_score',
                   'cv_variability_score', 'predictive_difficulty_score', 'predictive_difficulty_rank']:
        assert np.allclose(rerun[column], reference[column], atol=1e-10, rtol=0, equal_nan=True), column
    assert rerun.predictive_difficulty_label.equals(reference.predictive_difficulty_label)
    assert rerun.ranking_eligible.equals(reference.ranking_eligible)
    print('PASS: full fresh-kernel execution reproduces saved canonical predictive results', flush=True)

    destination = output / 'robustness'
    destination.mkdir(exist_ok=True)
    for artifact in (temporary / 'robustness').iterdir():
        if artifact.is_file():
            shutil.copy2(artifact, destination / artifact.name)

    # Retain original outputs outside the changed section, including timed burden.
    # Seed cells alone receive their verified new outputs.
    def real_output_paths(value):
        if isinstance(value, str):
            return value.replace(str(temporary), str(output))
        if isinstance(value, list):
            return [real_output_paths(v) for v in value]
        if isinstance(value, dict):
            return {k: real_output_paths(v) for k, v in value.items()}
        return value
    for before, executed in zip(original.cells, notebook.cells):
        if before.cell_type == 'code' and (before.source.startswith('RUN_SEED_ROBUSTNESS')
                                            or before.source.startswith('# SEED ROBUSTNESS:')):
            before.outputs = [nbformat.from_dict(real_output_paths(o)) for o in executed.outputs]
            before.execution_count = executed.execution_count
    nbformat.write(original, path)

assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in canonical_hashes.items())
print('PASS: all existing canonical output files remain byte-for-byte unchanged', flush=True)
print('NOTEBOOK AND SEED ROBUSTNESS VERIFIED SUCCESSFULLY', flush=True)
