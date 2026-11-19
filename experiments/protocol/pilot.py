"""Seven-fold Block 2A diagnostic selection. Defaults to a dry run."""
import argparse
import json
import logging

from .runner import Runner, plan


def pilot_tasks():
    common = dict(seeds=[42], repeats=[1], folds=[1])
    return (plan(datasets=['banknote'], models=['decision_tree_default', 'logistic_regression',
                                              'dummy_most_frequent', 'dummy_stratified'], **common)
            + plan(datasets=['car'], models=['svm_rbf', 'ripper'], **common)
            + plan(datasets=['iris'], models=['decision_tree_leaf_2_balanced'], **common))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--max-new-folds', type=int, default=7)
    args = parser.parse_args()
    if not 0 <= args.max_new_folds <= 7:
        parser.error('Pilot budget must be between 0 and 7')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    runner = Runner('block2a-seven-fold-pilot')
    result = runner.run(pilot_tasks(), execute=args.execute, resume=True, max_new_folds=args.max_new_folds)
    print(json.dumps(result, indent=2))
    return 1 if result['failed'] or result['corrupt'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
