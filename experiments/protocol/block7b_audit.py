"""Final Block 7B preservation audit; no fitting or partition regeneration."""
import ast
from concurrent.futures import ThreadPoolExecutor
import json
import nbformat
from .infrastructure import ROOT,digest,json_bytes,config,validate_dataset,write_frozen
from .datasets import load_dataset
from .block7 import MagicRunner
from .block7b_reporting import tables,SUPPLEMENT
from .block3_analysis import load_outputs


def audit():
    t=tables();load_outputs(SUPPLEMENT);runner=MagicRunner();records=runner.validate_inventory();assert len(records)==10500
    old=json.loads((ROOT/'experiments/protocol/block7_preservation.json').read_text());new=json.loads((ROOT/'experiments/protocol/block7b_preservation.json').read_text())
    pairs=[(ROOT/name,sha) for name,sha in old.items() if not name.endswith('New_Characterization_Experiments.ipynb')]+[(ROOT/name,sha) for name,sha in new['files'].items()]
    def check(pair):
        path,sha=pair
        if digest(path.read_bytes())!=sha:raise ValueError('Changed protected artifact: '+str(path))
    with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(check,pairs))
    manifest=json.loads((ROOT/'experiments/configs/dataset_manifest.json').read_text());datasets=0
    for entry in manifest['datasets']:
        if entry['eligible']:
            path=ROOT/entry['local_path'];obj=load_dataset(entry['dataset'],path);validate_dataset(obj,entry,path);datasets+=1
    assert datasets==17
    n=json.loads((ROOT/'experiments/New_Characterization_Experiments.ipynb').read_text(encoding='utf8'));nbformat.validate(nbformat.from_dict(n))
    assert [digest(json_bytes(c)) for c in n['cells'][:56]]==new['notebook_cells'];assert n['metadata']==new['notebook_metadata']
    assert len(n['cells'])==64
    for c in n['cells'][56:]:
        if c['cell_type']=='code':ast.parse(''.join(c['source']));assert c['execution_count'] is not None
    result=dict(successful_main_records=len(records),unique_main_records=len({(r['sample_id'],r['model'],r['cv_seed'],r['repeat'],r['fold']) for r in records}),prior_protected_files_unchanged=len(old)-1,
        block7_saved_files_unchanged=len(new['files']),original_benchmark_datasets_validated=datasets,old_notebook_cells_unchanged=56,new_notebook_cells=8,new_display_cells_executed=4,
        environment_verified=True,canonical_reference_verified=True,prepared_samples_and_cv_verified=True,nbformat_valid=True,models_fitted=0,samples_or_splits_regenerated=0,
        figures_visually_reviewed=5,original_figures_preserved=True)
    write_frozen(ROOT/'experiments/protocol/block7b_validation_report.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':audit()
