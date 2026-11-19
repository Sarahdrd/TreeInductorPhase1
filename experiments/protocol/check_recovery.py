"""Reproducible lightweight recovery audit; writes no scientific result records."""
import ast
import hashlib
import io
import json
from pathlib import Path
import unittest
import warnings
from importlib import metadata

from . import recovered_cross_models as recovered
from .infrastructure import ROOT, environment
from .test_recovered_models import RecoveryTests
from .test_infrastructure import InfrastructureTests


def main():
    suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(c)
                               for c in (RecoveryTests, InfrastructureTests)])
    output = io.StringIO()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
    original = json.loads((ROOT / 'experiments/configs/historical_cross_model_manifest.json').read_text())
    # Reuse the exact historical signature algorithm, never the evaluation loop.
    source = (ROOT / 'experiments/protocol/recovered_source/cross_model_cell_16.py.txt').read_text(encoding='utf-8')
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
             and n.name in ('cross_code_signature', 'cross_implementation')]
    namespace = dict(hashlib=hashlib, json=json)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<historical-signatures>', 'exec'), namespace)
    signature = namespace['cross_implementation']
    functions = {
        'svm_rbf': [recovered.build_preprocessor, recovered.make_svm_model, recovered.check_cross_predictions],
        'ripper': [recovered.RipperPreprocessor.fit, recovered.RipperPreprocessor.transform,
                   recovered.make_cross_model, recovered.check_cross_predictions,
                   recovered.SklearnRIPPER.__sklearn_tags__]}
    parameters = dict(svm_rbf=recovered.SVC(**recovered.SVM_CONFIG).get_params(),
                      ripper=recovered.SklearnRIPPER(random_state=42).get_params())
    comparisons = {}
    for name, params in parameters.items():
        old = original['model_configs'][name]
        differences = {k: dict(historical=old['estimator'].get(k), current=params.get(k))
                       for k in old['estimator'].keys() | params.keys()
                       if old['estimator'].get(k) != params.get(k)}
        current_hash = signature(functions[name])
        comparisons[name] = dict(parameter_differences=differences, current_parameters=params,
                                 historical_implementation_hash=old['implementation'],
                                 current_implementation_hash=current_hash,
                                 implementation_hash_matches=current_hash == old['implementation'])
    before = json.loads((ROOT / 'experiments/configs/environment.json').read_text())
    now = environment()
    changes = {k: dict(before=v, after=now['packages'][k]) for k, v in before['packages'].items()
               if v != now['packages'][k]}
    all_warnings = sorted({str(w.message) for w in caught} |
                          set((RecoveryTests.compatibility or {}).get('warnings', [])))
    report = dict(status='passed' if result.wasSuccessful() else 'failed', tests_run=result.testsRun,
                  failures=len(result.failures), errors=len(result.errors), test_output=output.getvalue(),
                  warnings=all_warnings, original_compatibility=RecoveryTests.compatibility,
                  historical_manifest_comparison=comparisons, environment=now,
                  package_changes=changes, install_command='uv --no-cache pip install --python .venv/Scripts/python.exe --no-deps wittgenstein==0.3.5',
                  installed_requires_dist=metadata.requires('wittgenstein'),
                  scope='10 small fixture pipeline fits: 2 original smoke, 6 RIPPER feature/class cases, 2 SVM; no full CV, benchmark evaluation or rankings',
                  svm_probability_difference='sklearn 1.9 sentinel deprecated resolves to effective False; factory unchanged',
                  historical_pairing='Still unverified for SVM/RIPPER: recovered source does not supply saved exact historical indices')
    destination = ROOT / 'experiments/protocol/recovery_validation_report.json'
    destination.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(output.getvalue())
    print(json.dumps(dict(status=report['status'], package_changes=changes,
                          comparison=comparisons, warning_types=len(all_warnings)), indent=2))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    main()
