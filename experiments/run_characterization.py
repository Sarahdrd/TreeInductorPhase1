"""Execute notebook in the current Python process when a Jupyter kernel is unavailable."""
import argparse
import base64
import os
import contextlib
import io
import time
from pathlib import Path
import nbformat
import matplotlib
matplotlib.use('Agg')
import IPython.display
p = Path(__file__).resolve().with_name('New_Characterization_Experiments.ipynb')
os.chdir(p.parent.parent)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--require-cache', action='store_true', help='Fail if any CV model fitting is attempted')
args = parser.parse_args()
if args.require_cache:
    import sklearn.model_selection
    def reject_fitting(*args, **kwargs):
        raise RuntimeError('Cache verification: unexpected model fitting')
    sklearn.model_selection.cross_validate = reject_fitting
n = nbformat.read(p, as_version=4)
namespace = {'__name__': '__main__'}
nbformat.validate(n)
for i,c in enumerate(n.cells):
    if c.cell_type == 'code':
        compile(c.source, f'characterization_cell_{i}', 'exec')
count = 0
for i,c in enumerate(n.cells):
    if c.cell_type != 'code':
        continue
    count += 1
    print(f'Executing cell {i}', flush=True)
    c.outputs = []
    c.execution_count = count
    def display(*objects, **kwargs):
        for obj in objects:
            data = {'text/plain': repr(obj)}
            if hasattr(obj, '_repr_html_'):
                data['text/html'] = obj._repr_html_()
            c.outputs.append(nbformat.v4.new_output('display_data', data=data))
    def show(*args, **kwargs):
        import matplotlib.pyplot as plt
        for number in plt.get_fignums():
            buffer = io.BytesIO()
            plt.figure(number).savefig(buffer, format='png', dpi=120, bbox_inches='tight')
            c.outputs.append(nbformat.v4.new_output('display_data', data={
                'image/png': base64.b64encode(buffer.getvalue()).decode('ascii'),
                'text/plain': 'Experiment figure (also exported as high-resolution PNG and PDF)'}))
    import matplotlib.pyplot as plt
    plt.show = show
    IPython.display.display = display
    namespace['display'] = display
    output = io.StringIO()
    started = time.perf_counter()
    try:
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            exec(compile(c.source, f'characterization_cell_{i}', 'exec'), namespace)
    except Exception as exc:
        import traceback
        c.outputs.append(nbformat.v4.new_output('error', ename=type(exc).__name__, evalue=str(exc), traceback=traceback.format_exc().splitlines()))
        print(output.getvalue(), flush=True)
        raise
    finally:
        if output.getvalue():
            c.outputs.insert(0,nbformat.v4.new_output('stream',name='stdout',text=output.getvalue()))
        nbformat.write(n,p)
    print(f'Cell {i} completed in {time.perf_counter()-started:.1f}s\n{output.getvalue()[-2000:]}',flush=True)
if args.require_cache:
    events = namespace['experiment_store'].events
    expected = sum(len(namespace['pool']) * k for k in (6,8))
    assert len(events) == expected and all(e['status'] == 'cache_hit' for e in events), events
    print(f'Cache-only verification: {expected} evaluations reused, zero fits.', flush=True)
print('Full notebook execution completed.',flush=True)
