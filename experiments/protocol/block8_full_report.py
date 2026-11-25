"""First-stage official coverage diagnostics only; no stratum-performance inference."""
import json
from pathlib import Path
import pandas as pd
from .block8_full import Production,BASE,MODELS,plan
from .infrastructure import ROOT,digest,json_bytes

OUT=ROOT/'experiments/analysis/block8-official-first-stage-v1'

def run():
    r=Production();rows=r.validate();expected=set(plan(True))
    keys={(v['dataset'],v['model'],v['seed'],v['repeat'],v['fold']) for v in rows}
    if keys!=expected or len(rows)!=68:raise ValueError(f'First-stage coverage is {len(rows)}/68 or contains other folds; do not publish incomplete stage')
    frame=pd.DataFrame(rows);frame['timed_out']=frame.termination_status.isin(['external_timeout','internal_timeout_no_tree'])
    frame['failed']=~frame.prediction_available & ~frame.timed_out
    coverage=frame.groupby('model').agg(evaluated=('dataset','size'),successful=('prediction_available','sum'),timed_out=('timed_out','sum'),failed=('failed','sum'),wall_mean=('wall_seconds','mean'),wall_max=('wall_seconds','max'),job_peak_max_bytes=('peak_job_memory_bytes','max')).reset_index()
    if OUT.exists():raise FileExistsError(OUT)
    OUT.mkdir()
    frame.drop(columns=['artifact_hashes']).to_csv(OUT/'first_stage_records.csv',index=False,float_format='%.17g')
    coverage.to_csv(OUT/'execution_coverage.csv',index=False,float_format='%.17g')
    frame[['dataset','model','termination_status','prediction_available','wall_seconds','peak_job_memory_bytes','validation_macro_f1']].to_csv(OUT/'feasibility_report.csv',index=False,float_format='%.17g')
    difficulties=frame[~frame.prediction_available]
    difficulties.drop(columns=['artifact_hashes']).to_csv(OUT/'execution_difficulties.csv',index=False,float_format='%.17g')
    provenance=dict(configuration_id=r.configuration_id,configuration_sha256=digest((BASE/'configuration.json').read_bytes()),records={str(p.relative_to(r.store.root)):digest(p.read_bytes()) for p in r.store.root.glob('records/**/attempt_*.json')},stage='seed 42 repetition 1 fold 1; one fold per dataset; not scientific performance evidence',remaining_unattempted=5100-68)
    (OUT/'provenance.json').write_bytes(json_bytes(provenance));(OUT/'output_checksums.json').write_bytes(json_bytes({p.name:digest(p.read_bytes()) for p in OUT.iterdir()}))
    def md(t):
        return '| '+' | '.join(t.columns)+' |\n| '+' | '.join(['---']*len(t.columns))+' |\n'+'\n'.join('| '+' | '.join(f'{v:.3f}' if isinstance(v,float) else str(v) for v in row)+' |' for row in t.itertuples(index=False,name=None))
    paragraphs=['# Block 8B — Official first-stage execution',
        '## Scope and immutable protocol',
        'Completed exactly 68 official evaluations: all 17 canonical datasets, both solvers and both representation-matched depth-2 CART configurations, using only seed 42, repetition 1, fold 1. The remaining 5,032 evaluations were not launched. Phase 8A pilots remain separate and are not copied into this new configuration. These official stage-one records are retained for full-run resume.',
        'Frozen configuration: `block8-optimal-trees-v1/official-depth2-v1/configuration.json`; configuration ID `'+r.configuration_id+'`. Settings: depth 2; MurTree at most 3 decision nodes, zero sparsity penalty, at most 16 numeric cuts per original feature and internal time limit 120 seconds; unchanged Quant-BnB native depth-2 settings; 120-second external wall and 4-GiB committed-memory job limits; serial execution and one numerical thread. The same external envelope is applied to separate CART worker processes. Preprocessing is training-only and occurs outside the solver/worker fit envelope, with its duration recorded separately. Total attempt time includes serialization, fitting and prediction checks.',
        'The original authors’ implementations and binaries, Julia project, canonical runtime, preprocessing/runner/worker/controller source hashes, dataset manifest, all 85 saved split collections, model parameters and canonical label reference are pinned. Resume refuses a changed configuration. No canonical constructor, dataset, split, score or previous artifact was modified.',
        '## Execution outcomes',md(coverage),
        f"Successful predictions: {int(frame.prediction_available.sum())}/68. Timeouts without usable predictions: {int(frame.timed_out.sum())}. Other failures: {int(frame.failed.sum())}.",
        'Macro-F1 is absent for unsuccessful fits. Quant-BnB predictions are feasible but uncertified. MurTree certification comes from the original solver flag; an external kill or missing returned tree never supplies a certificate. CART is recorded as a completed heuristic, not a globally optimal tree. All successful outputs pass label/length/depth/node validation, and solver training objectives are checked against returned training predictions.',
        '## Per-dataset coverage',md(frame.pivot(index='dataset',columns='model',values='termination_status').reset_index()),
        '## Runtime and memory interpretation',
        'wall_seconds measures external process startup through termination, including Julia startup/JIT and worker import overhead. solver_seconds, where supplied, measures the inner operation and is not directly interchangeable across languages. peak_job_memory_bytes is the Windows job peak committed memory, not Phase 8A’s polled working set. Memory allocation failures can yield nonzero exits; the controller does not infer a memory-limit cause from an arbitrary exit code. The 4-GiB limit is enforced for the attached job and descendants. Wall time has a small polling/cleanup overhead. Windows job kill-on-close handles interruptions and descendant cleanup. Logs and return codes are preserved per attempt.',
        '## Execution difficulties and coverage risk',
        md(difficulties[['dataset','model','termination_status','wall_seconds']]) if len(difficulties) else 'No first-stage execution difficulties were observed.',
        'One saved fold per dataset does not establish feasibility of the remaining 74. Different partitions may change preprocessing dimensions, search behavior or runtime; successes here are not a guarantee of complete 75-fold predictive coverage. Any committed failure or timeout remains an explicit missing predictive outcome. Subsequent analysis must distinguish attempted coverage from successful coverage and cannot aggregate incomplete datasets as if they had 75 valid scores. This stage makes no claim about stratum usefulness or comparative predictive superiority.',
        '## Resume and retry semantics',
        'A normal resume skips all committed successful, timed-out and failed outcomes. Interrupted attempts are automatically retried. Explicit --retry-failed appends a new attempt for unsuccessful outcomes; it never replaces history or reruns a successful fit. Changed budgets, source, environment or preprocessing require a new configuration version.',
        'After explicit authorization for the remaining folds, run from the repository root:\n\n```powershell\n.\\.venv\\Scripts\\python.exe -m experiments.protocol.block8_full resume --authorize-remaining\n```',
        'Read-only coverage validation:\n\n```powershell\n.\\.venv\\Scripts\\python.exe -m experiments.protocol.block8_full validate\n```',
        'Reissuing `first-stage` checks/skips its existing committed outcomes. It cannot expand to other folds. `resume` without `--authorize-remaining` is rejected. The full plan is 5,100 fits; the two solver configurations account for 2,550. Their worst-case external budget alone is 85 serial hours before overhead; CART/serialization adds time. This stage is complete and execution has stopped.',
        '## Artifacts and preservation',
        'Coverage, per-evaluation diagnostics and difficulties are in `analysis/block8-official-first-stage-v1/`, with checksums and record provenance. Full precision is retained. Work directories contain inputs, preprocessing, logs, native outputs and successful predictions, all linked by record hashes. No publication figures or notebook/manuscript edits were made. Validation results are saved separately in `block8-optimal-trees-v1/official-depth2-v1/first_stage_validation.json`.']
    (ROOT/'experiments/BLOCK8B_FIRST_STAGE.md').write_text('\n\n'.join(paragraphs)+'\n',encoding='utf8')
    print(coverage.to_string(index=False));print(difficulties[['dataset','model','termination_status']].to_string(index=False))

if __name__=='__main__':run()
