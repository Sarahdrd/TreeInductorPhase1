"""Read-only review validation; no data acquisition, fitting or split creation.

Works in a review-only checkout with published artifacts and notebook source
baselines. Display output is counted but not embedded back into notebooks.
"""
import ast
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from unittest.mock import patch

import nbformat
from .infrastructure import ROOT, digest, json_bytes
from .published_analysis import load_outputs


def validate():
    baseline=json.loads((ROOT/'experiments/repository_review/notebook_source_baseline.json').read_text())
    counts={}
    for name,ref in baseline.items():
        path=ROOT/'experiments'/name
        notebook=nbformat.read(path,as_version=4);nbformat.validate(notebook)
        if digest(json_bytes(notebook.metadata))!=ref['metadata_sha256']:
            raise AssertionError('Notebook metadata changed: '+name)
        for i,sha in enumerate(ref['source_sha256']):
            actual=digest(json_bytes(dict(cell_type=notebook.cells[i].cell_type,source=notebook.cells[i].source)))
            if actual!=sha and actual!=ref['authorized_display_import_sha256'].get(str(i)):
                raise AssertionError(f'Historical scientific source changed: {name} cell {i}')
        counts[name]=len(notebook.cells)
    output_tables={}
    for directory in (ROOT/'experiments/analysis').iterdir():
        if (directory/'published_checksums.json').is_file():
            output_tables[directory.name]=len(load_outputs(directory))
    n=nbformat.read(ROOT/'experiments/New_Characterization_Experiments.ipynb',as_version=4)
    display_calls=[];namespace={};executed=[]
    # Never execute the historical training cells (0--23).
    with patch('IPython.display.display',side_effect=lambda *a,**k:display_calls.append(len(a))), redirect_stdout(io.StringIO()):
        for i,cell in enumerate(n.cells):
            if i<24 or cell.cell_type!='code':continue
            tree=ast.parse(cell.source)
            for node in ast.walk(tree):
                if isinstance(node,ast.Call):
                    called=getattr(node.func,'attr',getattr(node.func,'id',''))
                    if called in {'fit','fit_transform','predict','run','generate_splits','nested_indices','execute'}:
                        raise AssertionError(f'Non-display call {called} in cell {i}')
            exec(compile(tree,f'analysis_cell_{i}','exec'),namespace)
            executed.append(i)
    t=load_outputs(ROOT/'experiments/analysis/block8-scientific-v1')
    labels=t['stratum_labels'].set_index('dataset')
    assert labels.loc['sonar','stratum_F']=='intermediate'
    assert labels.loc['parkinsons','stratum_F']=='hard'
    assert labels.loc['sonar','stratum_H']=='hard'
    assert labels.loc['parkinsons','stratum_H']=='intermediate'
    assert t['dataset_paired_deltas'].pairs.eq(75).all()
    assert t['solver_status'].fits.sum()==5100
    return dict(notebooks_valid=counts,original_source_and_metadata_preserved=True,
                authorized_changes='five display-loader imports/calls, cached outputs, appended Block 8 cells',
                published_table_counts=output_tables,analysis_cells_executed=executed,
                display_calls=len(display_calls),fits_or_solver_calls=0,F_H_boundary_checks=True)


if __name__=='__main__':print(json.dumps(validate(),indent=2))
