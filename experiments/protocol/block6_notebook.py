"""Append and execute only Block 6 read-only display cells."""
import base64
import json
from unittest.mock import patch
import nbformat
import pandas as pd
from IPython.display import Image
from .block6_analysis import load_outputs
from .infrastructure import ROOT,digest,json_bytes


def append_section():
    load_outputs()
    path=ROOT/'experiments/New_Characterization_Experiments.ipynb'
    n=json.loads(path.read_text(encoding='utf8'))
    protected=json.loads((ROOT/'experiments/protocol/block6_preservation.json').read_text())
    assert [digest(json_bytes(c)) for c in n['cells']]==protected['notebook_cells'],'Notebook changed or section already appended'
    assert n['metadata']==protected['notebook_metadata']
    cells=[]
    def add(kind,text):
        cell=dict(cell_type=kind,metadata={'tags':['block6-analysis-only']},source=text.splitlines(keepends=True),id=f'block6-{len(cells):02d}')
        if kind=='code':cell.update(outputs=[],execution_count=None)
        cells.append(cell)
    add('markdown','## Block 6 — DT-only characterization and paired regularization effects\n\n**Analysis only:** run these appended cells independently. They load checksummed saved exports and never fit models or regenerate folds. Earlier historical cells can train models. The mixed five-seed Block 3 canonical score and labels remain unchanged. All effect summaries use those labels; chance correction is excluded.\n')
    add('code',"""from pathlib import Path
import sys
import json
from IPython.display import display, Image
block6_root = next(p for p in [Path.cwd(), *Path.cwd().parents]
                   if (p / 'experiments/protocol/block6_analysis.py').is_file())
if str(block6_root) not in sys.path:
    sys.path.insert(0, str(block6_root))
from experiments.protocol.block6_analysis import load_outputs as load_block6_outputs
block6_output = block6_root / 'experiments/analysis/block6-tree-regularization-v1'
block6_tables = load_block6_outputs(block6_output)
block6_manifest = json.loads((block6_root / 'experiments/configs/dataset_manifest.json').read_text())
block6_names = {r['dataset']: r['canonical_name'] for r in block6_manifest['datasets']}
def block6_display(frame):
    shown = frame.copy()
    if 'dataset' in shown:
        shown['dataset'] = shown['dataset'].map(block6_names)
    display(shown.round(4))  # Display copy only; calculations and CSVs keep full precision.
block6_display(block6_tables['canonical_with_dt_only_assignments'])
block6_display(block6_tables['mixed_vs_dt_only_summary'])
""")
    add('markdown','### DT-only and regularization ranking sensitivity\n\nEach panel uses the original scoring, independently normalized on the same 17 datasets. Four stratum assignments change with unpruned DT-only characterization. Ranking agreement is not independent validation because DT contributes to the mixed score. No preferred pool or panel is selected.\n')
    add('code',"""block6_display(block6_tables['dt_lr_performance_diagnostic'])
block6_display(block6_tables['regularization_rank_agreement'])
""")
    add('markdown','### Fold-paired performance differences by canonical stratum\n\nAll 10,200 deltas match exact saved folds and class-weight settings. Positive delta means regularization improved validation macro-F1. Dataset means equally average the two tree settings and five seeds; stratum means equally weight datasets. The repeated folds are dependent. No new significance tests are computed. The Hard-stratum mean is negative for every panel, with heterogeneous individual effects and influential datasets.\n')
    add('code',"""block6_display(block6_tables['dataset_regularization_effects'])
block6_display(block6_tables['stratum_regularization_effects'])
block6_display(block6_tables['class_weight_comparison'])
block6_display(block6_tables['stratum_influence'])
""")
    add('markdown','### Publication figures and interpretation\n\nFigures are saved as 300-dpi PNGs and vector PDFs. Full provenance and paired records accompany the tables. See `BLOCK6_TREE_REGULARIZATION.md` for R1.3a–b/R1.4a–b evidence and the training-gap editorial assessment. Official training scores are absent; historical training values are discussed only with their separate provenance and are never substituted for current results. No manuscript or previous notebook sections were edited.\n')
    add('code',"""for block6_figure in ['mixed_vs_dt_only','regularization_rank_agreement','dataset_regularization_effects',
                       'stratum_regularization_effects','class_weight_sensitivity']:
    display(Image(filename=str(block6_output / (block6_figure + '.png')), width=1100))
""")
    ns={};number=0
    for cell in cells:
        if cell['cell_type']!='code': continue
        number+=1;cell['execution_count']=number
        def capture(value):
            metadata={}
            if isinstance(value,pd.DataFrame):
                data={'text/plain':value.to_string(index=False,float_format=lambda x:f'{x:.4f}'),
                      'text/html':value.to_html(index=False,float_format=lambda x:f'{x:.4f}')}
            elif isinstance(value,Image):
                payload=value._repr_png_()
                if isinstance(payload,tuple):payload=payload[0]
                data={'image/png':payload if isinstance(payload,str) else base64.b64encode(payload).decode('ascii'),'text/plain':'Publication figure'}
                metadata={'image/png':{'width':1100}}
            else:data={'text/plain':str(value)}
            cell['outputs'].append(dict(output_type='display_data',data=data,metadata=metadata))
        with patch('IPython.display.display',capture):exec(compile(''.join(cell['source']),cell['id'],'exec'),ns)
    n['cells'].extend(cells)
    nbformat.validate(nbformat.from_dict(n))
    assert [digest(json_bytes(c)) for c in n['cells'][:len(protected['notebook_cells'])]]==protected['notebook_cells']
    path.write_text(json.dumps(n,ensure_ascii=False,indent=1)+'\n',encoding='utf8')
    print(f'Preserved {len(protected["notebook_cells"])} cells; appended {len(cells)}; executed {number} new display cells only')


if __name__=='__main__':append_section()
