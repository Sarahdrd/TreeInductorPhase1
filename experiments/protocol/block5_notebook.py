"""Append and execute only Block 5 read-only display cells."""
import base64
import json
from unittest.mock import patch
import nbformat
import pandas as pd
from IPython.display import Image
from .block5_analysis import load_outputs
from .infrastructure import ROOT,digest,json_bytes


def append_section():
    load_outputs()
    path=ROOT/'experiments/New_Characterization_Experiments.ipynb'
    n=json.loads(path.read_text(encoding='utf8'))
    protected=json.loads((ROOT/'experiments/protocol/block5_preservation.json').read_text())
    assert [digest(json_bytes(c)) for c in n['cells']]==protected['notebook_cells'],'Notebook changed or section already appended'
    assert n['metadata']==protected['notebook_metadata']
    cells=[]
    def add(kind,text):
        cell=dict(cell_type=kind,metadata={'tags':['block5-analysis-only']},source=text.splitlines(keepends=True),id=f'block5-{len(cells):02d}')
        if kind=='code':cell.update(outputs=[],execution_count=None)
        cells.append(cell)
    add('markdown','## Block 5 — Label stability, weight sensitivity and stratum boundaries\n\n**Analysis only:** run these cells independently. They read checksummed saved exports, never fit models or regenerate CV. Earlier historical notebook cells may train models. All comparisons use the original Block 3 five-seed canonical labels at alpha 0.70. The chance-corrected Block 4 alternative does not enter this analysis.\n')
    add('code','''from pathlib import Path
import sys
from IPython.display import display, Image
block5_root = next(p for p in [Path.cwd(), *Path.cwd().parents]
                   if (p / 'experiments/protocol/block5_analysis.py').is_file())
if str(block5_root) not in sys.path:
    sys.path.insert(0, str(block5_root))
from experiments.protocol.block5_analysis import load_outputs as load_block5_outputs
block5_output = block5_root / 'experiments/analysis/block5-label-stability-v1'
block5_tables = load_block5_outputs(block5_output)
import json
block5_manifest = json.loads((block5_root / 'experiments/configs/dataset_manifest.json').read_text())
block5_names = {r['dataset']: r['canonical_name'] for r in block5_manifest['datasets']}
def block5_display(frame):
    shown = frame.copy()
    if 'dataset' in shown:
        shown['dataset'] = shown['dataset'].map(block5_names)
    display(shown.round(4))  # Presentation only; exported scores retain full precision.
block5_display(block5_tables['canonical_ranking_with_stability'])
''')
    add('markdown','### Agreement rates and scenario details\n\nSeed stability is matching labels out of five seeds at alpha 0.70. Weight stability is matching labels out of five aggregate alpha values (including 0.70). Combined stability is matching labels out of 25 seed × weight scenarios. Recommend the combined rate for the primary table, with the other rates in supplementary tables. These are related sensitivity scenarios, not independent replicates or probabilities of label correctness.\n')
    add('code','''for block5_table in ['seed_stability','weight_stability','combined_stability','rank_agreement_by_family']:
    block5_display(block5_tables[block5_table])
''')
    add('markdown','### Boundary membership and reviewer evidence\n\nSonar retains Intermediate in 21/25 scenarios (84%); Parkinsons retains Hard in 13/25 (52%). Eight datasets change label at least once. Both boundaries remain rank-based (6/7 and 12/13); plotted score midpoints are descriptive only. High Spearman correlation does not imply unchanged labels. See `BLOCK5_LABEL_STABILITY.md` for R1.2b–c evidence and a proposed abstract sentence; no manuscript text was edited.\n')
    add('code','''block5_display(block5_tables['boundary_analysis'])
block5_display(block5_tables['canonical_boundary_gaps'])
block5_display(block5_tables['boundary_exchanges'])
''')
    add('markdown','### Publication figures\n\nAll 300-dpi PNGs, vector PDFs, full-precision tables, scenario transitions and provenance are saved separately under `analysis/block5-label-stability-v1/`. Blocks 3 and 4 remain unchanged.\n')
    add('code','''for block5_figure in ['label_stability','scenario_heatmap','boundary_sensitivity','rank_vs_label_agreement']:
    display(Image(filename=str(block5_output / (block5_figure + '.png')), width=1100))
''')
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
