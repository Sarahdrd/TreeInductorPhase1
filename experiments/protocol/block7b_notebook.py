"""Append and execute only Block 7 read-only display cells."""
import base64
import json
from unittest.mock import patch
import nbformat
import pandas as pd
from IPython.display import Image
from .block7b_reporting import tables as load_outputs
from .infrastructure import ROOT,digest,json_bytes


def append_section():
    load_outputs()
    path=ROOT/'experiments/New_Characterization_Experiments.ipynb'
    n=json.loads(path.read_text(encoding='utf8'))
    protected=json.loads((ROOT/'experiments/protocol/block7b_preservation.json').read_text())
    assert [digest(json_bytes(c)) for c in n['cells']]==protected['notebook_cells'],'Notebook changed or section already appended'
    assert n['metadata']==protected['notebook_metadata']
    cells=[]
    def add(kind,text):
        cell=dict(cell_type=kind,metadata={'tags':['block7b-analysis-only']},source=text.splitlines(keepends=True),id=f'block7b-{len(cells):02d}')
        if kind=='code':cell.update(outputs=[],execution_count=None)
        cells.append(cell)
    add('markdown', '## Block 7 — MAGIC Subsampling\n\nAnalysis only: these cells load validated saved outputs; they never fit models or regenerate samples/folds. The complete main experiment has 10,500 successful evaluations. MAGIC remains separate from the 17-dataset canonical ranking.\n')
    add('code', """from pathlib import Path
import sys
from IPython.display import display, Image
block7_root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'experiments/protocol/block7b_reporting.py').exists())
if str(block7_root) not in sys.path:
    sys.path.insert(0, str(block7_root))
from experiments.protocol.block7b_reporting import tables, derived
block7_tables = tables()
block7_extra = derived(block7_tables)
block7_output = block7_root / 'experiments/analysis/block7-magic-subsampling-v1'
display(block7_tables['sample_size_summary'].round(6))
display(block7_extra['endpoint_changes'].round(6))
""")
    add('markdown', '### Sample-size sensitivity and distinct variability measures\n\nFrom 200 to 10,000 observations, mean macro-F1 rises 0.728212→0.778330; E falls 0.271788→0.221670; V falls 0.070221→0.009149 (86.97%); fixed-reference D_pred falls 0.521132→0.216789. Within-CV V averages 15-fold sample SDs across five CV seeds and four models. Between-subsample SD describes the five sample summaries. Between-CV-seed SD describes five seed-level mean scores; it is not V. Error bars show between-subsample SD, not confidence intervals. Fixed Block 3 endpoints are unchanged; no clipping or stratum assignment occurs.\n')
    add('code', """display(block7_tables['normalization_reference'])
display(block7_extra['variability_diagnostics'].round(6))
display(block7_tables['adjacent_size_consistency'].round(6))
display(block7_tables['replicate_trends'].round(6))
""")
    add('markdown', '### Monotonicity, diminishing returns and model differences\n\nV decreases in 29/30 within-sequence size changes; D_pred decreases in all 30. Seed 202 has a small V increase at 1,500→2,000. F improves in 24/30 comparisons. Mean F falls slightly at 1,000→1,500 while V and D_pred decrease: score improvement does not always mean improved prediction. Changes per added observation diminish overall, with irregular F behavior; no smooth curve or plateau is imposed. DT mean F improves at every size, whereas both LR means peak at 1,000 in this grid and decline slightly at the final transition. All four mean V curves decrease throughout.\n')
    add('code', """display(block7_extra['model_size_summary'].round(6))
display(block7_extra['adjacent_size_rates'].round(6))
display(block7_extra['score_change_decomposition'].round(6))
""")
    add('markdown', '### Figures and reviewer interpretation\n\nGray trajectories are five nested subsampling sequences; blue points are means and bars are ±1 between-sequence SD. X axes use log sample size; values are unitless. Model panels have different y limits. R2.2 and previous R2.5: these results support substantial finite-sample sensitivity of V for MAGIC and the fixed learners. Training information and validation-fold size change together, so the experiment cannot identify their separate contributions or prove V measures size alone. Sizes are nested, samples overlap, folds are dependent, and only one underlying dataset is studied. This does not remove size confounding across the original benchmark or alter its canonical score. See BLOCK7_MAGIC_SUBSAMPLING.md for full provenance and limitations.\n')
    add('code', """for block7_figure in ['sample_size_error','sample_size_variability','sample_size_difficulty_score','joint_components','model_specific_trends']:
    display(Image(filename=str(block7_output / (block7_figure + '.png')), width=1100))
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
