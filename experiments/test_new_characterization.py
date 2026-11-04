"""Integrity regressions for the characterization notebook and its JSON cache."""
import ast
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import nbformat
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.characterization_persistence import ExperimentStore, validate_evaluations, code_digest

NOTEBOOK = Path(__file__).with_name('New_Characterization_Experiments.ipynb')
ns = {}
notebook = nbformat.read(NOTEBOOK, as_version=4)
exec(notebook.cells[3].source, ns)
# Extract shared functions without executing any model fitting or exports.
for index in (9, 12, 15):
    tree = ast.parse(notebook.cells[index].source)
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if index == 9:
        nodes += [node for node in tree.body if isinstance(node, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id in {
                      'CV_N_SPLITS','N_JOBS','NON_DUMMY_BASELINE_MODELS','TREE_BASELINE_MODELS',
                      'DUMMY_REFERENCE_MODELS','TREE_CONFIG','LR_CONFIG','MODEL_CONFIGS'} for t in node.targets)]
    exec(compile(ast.Module(body=sorted(nodes,key=lambda x:x.lineno), type_ignores=[]), 'test_functions', 'exec'), ns)
ns.update(validate_evaluations=validate_evaluations, GROUPS=['easy','intermediate','hard'])


def row(dataset='d', model='logistic_regression', offset=0.):
    scores = np.linspace(.4,.8,15) + offset
    return dict(dataset=dataset, model=model, cv_seed=42, cv_folds=5, cv_repeats=3,
                n_scores=15, scores=scores.tolist(), error='', warnings='',
                f1_macro_mean=scores.mean(), f1_macro_std=scores.std(ddof=1))


class IntegrityTests(unittest.TestCase):
    def test_duplicate_missing_seed_nonfinite_and_summary_rejected(self):
        good = pd.DataFrame([row()])
        validate_evaluations(good, ['d'], ['logistic_regression'])
        invalid = [pd.concat([good,good]), good.iloc[:0]]
        for field,value in [('cv_seed',43),('n_scores',14),('error','fit failed'),('f1_macro_std',.99)]:
            invalid.append(good.assign(**{field:value}))
        bad = copy.deepcopy(row()); bad['scores'][0] = float('nan')
        invalid.append(pd.DataFrame([bad]))
        for frame in invalid:
            with self.subTest(frame=frame.to_dict('records')):
                with self.assertRaises(ValueError):
                    validate_evaluations(frame,['d'],['logistic_regression'])

    def test_variability_is_mean_within_model_std(self):
        frame = pd.DataFrame([row(model='a'),row(model='b',offset=.15)])
        audit = pd.DataFrame([dict(dataset='d',comparable_cv=True,min_class_count=5,cv_issue='')])
        result = ns['aggregate_baselines'](frame,audit,['a','b']).iloc[0]
        self.assertAlmostEqual(result.cv_variability, np.std(row()['scores'],ddof=1))
        self.assertNotAlmostEqual(result.cv_variability,np.std(frame.scores.sum(),ddof=1))
        with self.assertRaises(ValueError):
            ns['aggregate_baselines'](pd.concat([frame,frame.iloc[:1]]),audit,['a','b'])

    def test_exact_ties_and_17_dataset_group_sizes(self):
        names = [f'd{i:02}' for i in range(17)][::-1]
        components = pd.DataFrame(dict(dataset=names,F=.8,E=.2,V=.05,normalized_E=0.,normalized_V=0.))
        panel = ns['score_panel'](components,'canonical')
        self.assertEqual(panel.sort_values('position').dataset.tolist(),sorted(names))
        self.assertEqual(panel.difficulty_group.value_counts().reindex(ns['GROUPS']).tolist(),[6,6,5])
        rho, _, status = ns['safe_spearman'](panel.difficulty_score,panel.position)
        self.assertTrue(np.isnan(rho)); self.assertIn('constant',status)
        rho,_,_ = ns['safe_spearman']([0,0,1,2],[0,1,1,2])
        self.assertAlmostEqual(rho,5/6)
        with self.assertRaises(ValueError): ns['safe_spearman']([0,np.nan],[1,2])

    def test_fixed_pool_failure_and_recomputed_dt_normalization(self):
        ns['pool'] = ['a','b','c']
        models = ['logistic_regression','logistic_regression_balanced']
        rows = [row(d,m,i*.05) for i,d in enumerate(ns['pool']) for m in models]
        ns['cv_audit_df'] = pd.DataFrame([dict(dataset=d,comparable_cv=True,min_class_count=5,cv_issue='') for d in ns['pool']])
        result = ns['panel_components'](pd.DataFrame(rows),models)
        np.testing.assert_allclose(result.normalized_E,[1,.5,0])
        rows[0]['error']='failure'
        with self.assertRaises(ValueError): ns['panel_components'](pd.DataFrame(rows),models)

    def test_cache_roundtrip_rejection_and_split_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            obj = dict(X=pd.DataFrame({'x':range(10)}),y=np.array([0]*5+[1]*5),target_col='target',dropped_id_cols=[])
            source=root/'data.csv'; obj['X'].to_csv(source,index=False)
            store=ExperimentStore(root/'output',{'d':obj},{'d':source},[row],{'d':'target'},{'d':[]},[])
            estimator=ns['LogisticRegression']()
            signature,spec=store.signature('d','logistic_regression',estimator,1)
            store.save(row(),signature,spec)
            self.assertEqual(store.read('d','logistic_regression',signature)['scores'],row()['scores'])
            self.assertIsNone(store.read('d','logistic_regression','wrong'))
            self.assertEqual(len(store.splits['d']),15)
            for repeat in range(3):
                tests=np.concatenate([test for _,test in store.splits['d'][repeat*5:(repeat+1)*5]])
                np.testing.assert_array_equal(np.sort(tests),np.arange(10))
            changed,_=store.signature('d','logistic_regression',ns['LogisticRegression'](C=2),1)
            self.assertNotEqual(signature,changed)
            entry=store.cache/'d__logistic_regression.json'
            payload=json.loads(entry.read_text()); payload['row']['scores'][0]=.99
            entry.write_text(json.dumps(payload))
            self.assertIsNone(store.read('d','logistic_regression',signature))
            relocated=type(row)(row.__code__.replace(co_filename='another execution',co_firstlineno=99),row.__globals__)
            self.assertEqual(code_digest(row),code_digest(relocated))

if __name__=='__main__': unittest.main(verbosity=2)
