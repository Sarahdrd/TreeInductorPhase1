"""Execute the notebook from a fresh kernel and save outputs, including failures."""
import os
import sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpec

root=Path(__file__).resolve().parents[1]
path=root/'experiments'/'sensitivity_ranking_test.ipynb'
os.environ['MPLBACKEND']='module://matplotlib_inline.backend_inline'
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
nb=nbformat.read(path,as_version=4)
for i,c in enumerate(nb.cells):
    if c.cell_type=='code': compile(c.source,f'cell {i}','exec')
km=KernelManager()
km._kernel_spec=KernelSpec(argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],display_name='Python',language='python')
def progress(cell,cell_index,**kwargs):
    print(f'Executing cell {cell_index}: {cell.source[:80]!r}',flush=True)
client=NotebookClient(nb,km=km,timeout=1800,resources={'metadata':{'path':str(root)}},on_cell_start=progress)
try:
    client.execute()
finally:
    nbformat.write(nb,path)
print('NOTEBOOK EXECUTED SUCCESSFULLY',flush=True)
