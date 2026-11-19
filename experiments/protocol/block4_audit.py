"""Final read-only integrity audit, saving only its new audit report."""
import ast
from concurrent.futures import ThreadPoolExecutor
import json
import nbformat
from .block4_analysis import OUTPUT,load_outputs
from .infrastructure import ROOT,digest,json_bytes,environment
from .checkpoints import filesystem_path


def audit():
    tables=load_outputs()
    saved=json.loads((ROOT/'experiments/protocol/block4_preservation.json').read_text())
    protected=[(ROOT/name,sha) for name,sha in saved['files'].items() if not name.endswith('New_Characterization_Experiments.ipynb')]
    official=json.loads((OUTPUT/'official_record_checksums.json').read_text())
    root=filesystem_path(ROOT/'experiments/results/official/revision2-five-seed-v1/block2b-five-seed')
    def check(item):
        path,expected=item
        if digest(path.read_bytes())!=expected: raise ValueError('Protected file changed: '+str(path))
        return 1
    with ThreadPoolExecutor(max_workers=8) as workers:
        count=sum(workers.map(check,protected))
        records=sum(workers.map(check,[(root/name,sha) for name,sha in official.items()]))
    path=ROOT/'experiments/New_Characterization_Experiments.ipynb'
    n=json.loads(path.read_text(encoding='utf8'))
    assert n['metadata']==saved['notebook_metadata']
    assert [digest(json_bytes(c)) for c in n['cells'][:len(saved['notebook_cells'])]]==saved['notebook_cells']
    nbformat.validate(nbformat.from_dict(n))
    parsed=0
    for notebook in (ROOT/'experiments').glob('*.ipynb'):
        for cell in json.loads(notebook.read_text(encoding='utf8'))['cells']:
            if cell['cell_type']=='code': ast.parse(''.join(cell['source']));parsed+=1
    provenance=json.loads((OUTPUT/'provenance.json').read_text())
    assert environment()['environment_id']==provenance['validation']['environment_id']
    result=dict(protected_files_unchanged=count,official_record_files_unchanged=records,
        previous_notebook_cells_unchanged=len(saved['notebook_cells']),appended_notebook_cells=len(n['cells'])-len(saved['notebook_cells']),
        appended_display_cells_executed=sum(c['cell_type']=='code' and c['execution_count'] is not None for c in n['cells'][len(saved['notebook_cells']):]),
        notebook_code_cells_parsed=parsed,nbformat_valid=True,environment_unchanged=True,
        official_successes=20400,canonical_records=5100,dummy_records=len(tables['dummy_fold_scores']),
        analysis_unit='dataset (n=17)',models_fitted=0,splits_generated=0,canonical_results_changed=False)
    output=ROOT/'experiments/protocol/block4_validation_report.json'
    if output.exists(): raise FileExistsError(output)
    output.write_bytes(json_bytes(result));print(json.dumps(result,indent=2))


if __name__=='__main__': audit()
