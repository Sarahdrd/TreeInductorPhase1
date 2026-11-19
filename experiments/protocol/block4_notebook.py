"""Append independently runnable display cells; preserve all existing notebook cells."""
import contextlib
import io
import json
from unittest.mock import patch
import nbformat
import pandas as pd
from IPython.display import Image
from .block4_analysis import load_outputs
from .infrastructure import ROOT,digest,json_bytes


def append_section():
    load_outputs()  # Require complete, checksummed exports first.
    path=ROOT/'experiments/New_Characterization_Experiments.ipynb'
    n=json.loads(path.read_text(encoding='utf8'))
    saved=json.loads((ROOT/'experiments/protocol/block4_preservation.json').read_text())
    if len(n['cells'])!=len(saved['notebook_cells']): raise ValueError('Notebook changed or section already appended')
    assert [digest(json_bytes(c)) for c in n['cells']]==saved['notebook_cells']
    assert n['metadata']==saved['notebook_metadata']
    cells=[]
    def md(text): cells.append(dict(cell_type='markdown',metadata={'tags':['block4-analysis-only']},source=text.splitlines(keepends=True)))
    def code(text): cells.append(dict(cell_type='code',metadata={'tags':['block4-analysis-only']},source=text.splitlines(keepends=True),outputs=[],execution_count=None))
    md('## Block 4 — Chance-corrected predictive difficulty (comparative sensitivity)\n\nRun these appended cells independently. They read validated saved outputs and never evaluate models. Earlier notebook sections may fit models; do not use Run All for this analysis. The historical canonical score and Block 3 five-seed canonical score remain unchanged. This alternative uses stratified dummy macro-F1 and changes only the aggregate predictive-error component.\n')
    code('''from pathlib import Path
import sys
from IPython.display import display, Image
block4_root = next(p for p in [Path.cwd(), *Path.cwd().parents]
                   if (p / 'experiments/protocol/block4_analysis.py').is_file())
if str(block4_root) not in sys.path:
    sys.path.insert(0, str(block4_root))
from experiments.protocol.block4_analysis import load_outputs as load_block4_outputs
block4_output = block4_root / 'experiments/analysis/block4-chance-correction-v1'
block4_tables = load_block4_outputs(block4_output)
def block4_display(frame):
    display(frame.round(4))  # Display copy only: full precision is retained in exports.
block4_display(block4_tables['dummy_baselines'])
''')
    md('### Original versus corrected rankings\n\nE_cc = (1 − F)/(1 − F_dummy), using aggregate five-seed means. D_cc = 0.70 minmax(E_cc) + 0.30 V_norm, with unchanged Block 3 variability and the same 17-dataset normalization pool, ties and 6/6/5 strata. Both dummy strategies are reported, but only stratified dummy is the correction reference.\n')
    code('''block4_display(block4_tables['original_vs_corrected'][[
    'dataset_name','original_mean_macro_f1','stratified_dummy_macro_f1',
    'original_error','chance_corrected_error','original_difficulty','chance_corrected_difficulty',
    'original_rank','corrected_rank','rank_displacement','original_stratum','corrected_stratum']])
block4_display(block4_tables['rank_comparison_summary'])
''')
    md('### Class structure, Heart Disease and Wine Quality Red\n\nThese are exploratory associations across 17 datasets, not independent fold-level observations. Correction does not causally eliminate imbalance or metric effects. Absolute and normalized improvement over dummy are diagnostics, not additional score components. See `BLOCK4_CHANCE_CORRECTION.md` for the numerical interpretation and R1.1a–c traceability.\n')
    code('''block4_display(block4_tables['descriptor_correlations'])
block4_display(block4_tables['heart_wine_focus'])
block4_display(block4_tables['improvement_by_original_stratum'])
''')
    md('### Publication figures\n\nPNG figures are 300 dpi; vector PDFs and full-precision tables are in `analysis/block4-chance-correction-v1/`. These local exports may not all be tracked by Git. No preferred score is selected here.\n')
    code('''for block4_figure in ['rank_comparison','score_comparison','class_structure','dummy_vs_canonical']:
    display(Image(filename=str(block4_output / (block4_figure + '.png')), width=1000))
''')
    ns={};counter=0
    for i,cell in enumerate(cells):
        cell['id']=f'block4-{i:02d}'
        if cell['cell_type']!='code': continue
        counter+=1;cell['execution_count']=counter
        def display(value):
            if isinstance(value,pd.DataFrame):
                with pd.option_context('display.float_format',lambda x:f'{x:.4f}'):
                    data={'text/plain':value.to_string(index=False),'text/html':value.to_html(index=False,float_format=lambda x:f'{x:.4f}')}
            elif isinstance(value,Image):
                payload=value._repr_png_()
                if isinstance(payload,tuple): payload=payload[0]
                import base64
                data={'image/png':payload if isinstance(payload,str) else base64.b64encode(payload).decode('ascii'),'text/plain':'Publication figure'}
            else: data={'text/plain':str(value)}
            cell['outputs'].append(dict(output_type='display_data',data=data,metadata={'image/png':{'width':1000}} if isinstance(value,Image) else {}))
        with patch('IPython.display.display',display): exec(compile(''.join(cell['source']),f'block4-cell-{i}','exec'),ns)
    n['cells'].extend(cells)
    nbformat.validate(nbformat.from_dict(n))
    assert [digest(json_bytes(c)) for c in n['cells'][:len(saved['notebook_cells'])]]==saved['notebook_cells']
    path.write_text(json.dumps(n,ensure_ascii=False,indent=1)+'\n',encoding='utf8')
    print(f'Preserved {len(saved["notebook_cells"])} cells; appended {len(cells)}; executed only {counter} display cells')


if __name__=='__main__': append_section()
