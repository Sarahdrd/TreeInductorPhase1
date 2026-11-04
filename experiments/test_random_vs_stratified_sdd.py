"""Small deterministic and spawned-worker checks; never executes the main plan."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evolution import random_vs_stratified_sdd as ex
import numpy as np
import pandas as pd


def completed(mean=.5):
    return dict(status="ok", folds=[dict(fold=i, status="ok", macro_f1=mean) for i in range(5)])


class FakeEvaluator:
    """Deterministic synthetic evaluator for GA continuation, never for reported results."""
    def __init__(self, path, fail=False):
        self.cache, self.events, self.fail = Path(path), [], fail

    def evaluate(self, genes, names):
        results = {}
        for gene in genes:
            key = tuple(gene)
            results[key] = {name: completed(.3 + ((sum(gene) + i * gene[0]) % 50) / 100)
                            for i, name in enumerate(names)}
            if self.fail:
                results[key][names[0]] = dict(status="timeout", folds=[])
        fitness = [-(ex.aggregate(results[tuple(g)], names)["sdd"]) if not self.fail else -np.inf for g in genes]
        return fitness, results, dict(candidates_requested=len(genes), batch_unique_genes=len(results),
            duplicate_positions=len(genes) - len(results), cache_hits=0,
            dataset_requests=len(genes)*len(names), unique_dataset_requests=len(results)*len(names))


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temp.name)
        cls.protocol = ex.prepare(cls.output)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_sdd_percentage_and_incomplete(self):
        self.assertAlmostEqual(ex.sdd([0., 1.]), np.sqrt(20000))
        self.assertAlmostEqual(ex.sdd([.5, 1.]), (200/3)/np.sqrt(2))
        self.assertEqual(ex.sdd([0., 0., 0.]), 0.)  # Why macro-F1 must accompany SDD.
        for values in ([.8], [], [.5, np.nan], [-.1, .5]):
            with self.assertRaises(ValueError):
                ex.sdd(values)
        incomplete = {"a": completed(), "b": dict(status="timeout", folds=[])}
        self.assertIsNone(ex.aggregate(incomplete, ["a", "b"])["sdd"])
        self.assertIsNone(ex.aggregate({"a": completed()}, ["a", "b"])["macro_f1"])

    def test_prepare_preserves_saved_numeric_identity_and_resume(self):
        # argparse leaves the numeric default as int, but parses an explicit flag
        # as float. Equal protocols must return the same persisted hash in both cases.
        for initial, resumed in [(900, 900.0), (900.0, 900)]:
            with self.subTest(initial=type(initial).__name__), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                first = ex.prepare(root, config=ex.Config(gen_dataset_timeout_s=initial))
                selection = first["selections"][0]
                state = ex.run_search(first, selection, root, 0, FakeEvaluator(root))
                checkpoint = root / "runs" / selection["run_id"] / "checkpoint.json"
                checkpoint_bytes = checkpoint.read_bytes()
                again = ex.prepare(root, config=ex.Config(gen_dataset_timeout_s=resumed))
                self.assertEqual(ex.digest(first), ex.digest(again))
                before = ex.evaluation_spec([0]*8, first["datasets"]["haberman"], first)
                after = ex.evaluation_spec([0]*8, again["datasets"]["haberman"], again)
                self.assertEqual(ex.digest(before), ex.digest(after))
                with patch.object(ex.Evaluator, "evaluate", side_effect=AssertionError("Repeated initial population")):
                    restored = ex.run_search(again, again["selections"][0], root, 0)
                self.assertEqual(restored["signature"], state["signature"])
                self.assertEqual(checkpoint.read_bytes(), checkpoint_bytes)
                with self.assertRaisesRegex(ValueError, "Incompatible existing artifact"):
                    ex.prepare(root, config=ex.Config(gen_dataset_timeout_s=901))

    def test_pool_reserve_ranking_and_equal_budgets(self):
        protocol = self.protocol
        manifest = protocol["manifest"]
        self.assertEqual(len(manifest["eligible"]), 17)
        self.assertEqual(len(manifest["held_out"]), 5)
        self.assertFalse(set(manifest["available"]) & set(manifest["held_out"]))
        ranking = pd.read_csv(self.output / "available_pool_ranking.csv")
        self.assertEqual(set(ranking.dataset), set(manifest["available"]))
        self.assertEqual(ranking.difficulty_group.value_counts().to_dict(), dict(easy=4, intermediate=4, hard=4))
        self.assertEqual(ex.selections(manifest["available"], ranking), protocol["selections"])
        for row in protocol["selections"]:
            self.assertEqual(len(set(row["datasets"])), 3)
            self.assertTrue(set(row["datasets"]) <= set(manifest["available"]))
            if row["condition"] == "stratified":
                self.assertEqual(set(ranking.set_index("dataset").loc[row["datasets"], "difficulty_group"]), set(ex.GROUPS))
        self.assertEqual(ex.Config().mu + ex.Config().lambd * ex.Config().generations, 110)
        with self.assertRaises(ValueError):
            ex.Config(lambd=25)
        with self.assertRaises(ValueError):
            ex.Config(search_cv_folds=3)
        # Ranking cannot change with any held-out score because it sees only available rows.
        extra = pd.DataFrame([dict(dataset=n, F=0., E=1., V=999.) for n in manifest["held_out"]])
        original = ex.rank_available(ranking, manifest["available"])
        changed = ex.rank_available(pd.concat([ranking, extra]), manifest["available"])
        pd.testing.assert_frame_equal(original, changed)

    def test_targets_ids_missing_and_fold_preprocessing(self):
        datasets = self.protocol["datasets"]
        self.assertEqual(datasets["parkinsons"]["identity"]["resolved_ids"], ["name"])
        self.assertEqual(datasets["cylinder_bands"]["identity"]["resolved_ids"], ["feature_001"])
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / "iris.csv"
            pd.DataFrame(dict(feature=[1, 2, 3], wrong=[0, 1, 1])).to_csv(file, index=False)
            with self.assertRaisesRegex(ValueError, "target"):
                ex.load_data("iris", file)
            pd.DataFrame(dict(feature=[1, 2, 3], target=["a", None, "b"])).to_csv(file, index=False)
            obj, audit = ex.load_data("iris", file)
            self.assertEqual(len(obj["y"]), 2)
            self.assertEqual(audit["n_missing_targets_removed"], 1)
        train, valid = ex.fold_features(pd.DataFrame(dict(x=["b", "a", None], v=[1., np.nan, 3.])),
                                       pd.DataFrame(dict(x=["c", "a"], v=[np.nan, 2.])))
        self.assertTrue(np.isnan(train[2, 0]) and np.isnan(train[1, 1]) and np.isnan(valid[0, 0]))
        self.assertEqual(valid[1, 0], 0.)

    def test_stable_splits_and_cache_identity(self):
        dataset = self.protocol["datasets"]["iris"]
        obj, _ = ex.load_data("iris", dataset["identity"]["path"])
        self.assertEqual(ex.partitions(obj["y"], "iris"), (dataset["split_seed"], dataset["splits"]))
        with self.assertRaises(ValueError):
            ex.partitions(np.array([0] * 10 + [1] * 4), "rare")
        spec = ex.evaluation_spec([0]*8, dataset, self.protocol)
        for field, value in [("timeout_s", 4), ("preprocessing", "other"), ("model_seeds", [0]*5)]:
            changed = copy.deepcopy(spec); changed[field] = value
            self.assertNotEqual(ex.digest(spec), ex.digest(changed))

    def test_resume_rng_population_and_budget(self):
        selection = self.protocol["selections"][0]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            ex.run_search(self.protocol, selection, root/"resume", 1, FakeEvaluator(root))
            resumed = ex.run_search(self.protocol, selection, root/"resume", 3, FakeEvaluator(root))
            uninterrupted = ex.run_search(self.protocol, selection, root/"straight", 3, FakeEvaluator(root))
            for field in ("population", "fitness", "best_gene", "rng", "seen_genes"):
                self.assertEqual(resumed[field], uninterrupted[field], field)
            self.assertEqual(sum(h["candidates_requested"] for h in resumed["history"]), 40)
            paired = ex.run_search(self.protocol, self.protocol["selections"][1], root/"paired", 0, FakeEvaluator(root))
            self.assertEqual(resumed["initial_population"], paired["initial_population"])
            changed = copy.deepcopy(self.protocol); changed["config"]["mutation_prob"] = .2
            with self.assertRaisesRegex(ValueError, "Checkpoint protocol changed"):
                ex.run_search(changed, selection, root/"resume", 4, FakeEvaluator(root))

    def test_failure_never_wins_and_final_is_guarded(self):
        with tempfile.TemporaryDirectory() as folder:
            state = ex.run_search(self.protocol, self.protocol["selections"][0], folder, 2, FakeEvaluator(folder, fail=True))
            self.assertEqual(state["status"], "failed")
            self.assertIsNone(state["best_gene"])
            with patch.object(ex, "Evaluator", side_effect=AssertionError("Must not evaluate held out")):
                with self.assertRaisesRegex(ValueError, "All ten searches"):
                    ex.final_evaluation(self.protocol, folder)
        evaluator = ex.Evaluator(self.protocol, self.output)
        with self.assertRaisesRegex(ValueError, "Forbidden"):
            evaluator.evaluate([[0]*8], [self.protocol["manifest"]["held_out"][0]])

    def test_pilot_never_calls_final(self):
        # Run the actual pilot orchestration with synthetic search-only evaluator.
        original = ex.run_search
        with tempfile.TemporaryDirectory() as folder:
            def search(protocol, selection, output, target_generation):
                self.assertTrue(set(selection["datasets"]).isdisjoint(protocol["manifest"]["held_out"]))
                return original(protocol, selection, output, target_generation, FakeEvaluator(output))
            with patch.object(ex, "prepare", return_value=self.protocol), \
                 patch.object(ex, "run_search", side_effect=search), \
                 patch.object(ex, "final_evaluation", side_effect=AssertionError("Held-out evaluation during pilot")), \
                 patch.object(ex, "figures"):
                ex.execute("pilot", folder)
            self.assertFalse((Path(folder) / "final_run_summary.csv").exists())

    def test_real_spawn_cache_equivalence_dedup_and_timeout(self):
        # Synthetic files only: infrastructure validation must not train on the reserve.
        protocol = copy.deepcopy(self.protocol)
        protocol["manifest"]["available"] = ["haberman", "parkinsons"]
        protocol["manifest"]["held_out"] = []
        gene = [0, 0, 1, 0, 0, 0, 0, 1]
        with tempfile.TemporaryDirectory() as folder:
            for name, target in [("haberman", "3"), ("parkinsons", "status")]:
                file = Path(folder) / (name + ".csv")
                frame = pd.DataFrame({"feature": np.arange(30), target: [0, 1] * 15})
                if name == "parkinsons":
                    frame["name"] = [f"id_{i}" for i in range(30)]
                frame.to_csv(file, index=False)
                obj, audit = ex.load_data(name, file)
                split_seed, splits = ex.partitions(obj["y"], name)
                protocol["datasets"][name] = dict(name=name, identity=ex.identity(name, file, obj, audit),
                    audit=audit, split_seed=split_seed, splits=splits, splits_sha256=ex.digest(splits))
            names = ["haberman", "parkinsons"]
            evaluator = ex.Evaluator(protocol, folder)
            fitness, first, counts = evaluator.evaluate([gene, gene], names)
            result = first[tuple(gene)]["haberman"]
            self.assertTrue(ex.complete_result(result), result)
            self.assertEqual(counts["batch_unique_genes"], 1)
            self.assertEqual(sum(e["fits_started"] for e in evaluator.events), 10)
            cached_fitness, cached, counts = evaluator.evaluate([gene], names)
            self.assertEqual(counts["cache_hits"], 2)
            uncached = ex.Evaluator(protocol, folder, use_cache=False)
            fresh_fitness, fresh, _ = uncached.evaluate([gene], names)
            self.assertEqual(fitness, cached_fitness * 2)
            self.assertEqual(cached_fitness, fresh_fitness)
            for value in (cached, fresh):
                self.assertEqual([f["macro_f1"] for f in result["folds"]],
                                 [f["macro_f1"] for f in value[tuple(gene)]["haberman"]["folds"]])
            spec = ex.evaluation_spec(gene, protocol["datasets"]["haberman"], protocol)
            spec["timeout_s"] = .001
            timed = ex.timed_dataset(spec)
            self.assertEqual(timed["status"], "timeout")
            self.assertFalse(ex.complete_result(timed))

    def test_final_failure_coverage_fold_slots_and_paired_exports(self):
        protocol = self.protocol
        names = protocol["manifest"]["held_out"]
        winners = [dict(selection=s, gene=[i]*8) for i, s in enumerate(protocol["selections"])]
        class FinalStub:
            def __init__(self, *args, **kwargs):
                pass
            def evaluate(self, genes, requested):
                self_requested = list(requested)
                if self_requested != names:
                    raise AssertionError("Final reserve changed")
                gene = genes[0]
                result = {name: {**completed(.4 + gene[0] / 100), "wall_seconds": 1.} for name in requested}
                if gene[0] == 0:
                    result[names[0]] = dict(status="timeout", folds=[], wall_seconds=1.)
                return [0], {tuple(gene): result}, {}
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(ex, "freeze_winners", return_value=winners), patch.object(ex, "Evaluator", FinalStub):
                summary = ex.final_evaluation(protocol, folder)
                # Saved failures are retained; a second call must not evaluate any gene again.
                with patch.object(FinalStub, "evaluate", side_effect=AssertionError("Repeated final evaluation")):
                    repeated = ex.final_evaluation(protocol, folder)
            pd.testing.assert_frame_equal(summary, repeated)
            self.assertEqual(summary.iloc[0].coverage, 4)
            self.assertTrue(pd.isna(summary.iloc[0].macro_f1))
            self.assertTrue(pd.isna(summary.iloc[0].sdd))
            self.assertEqual(len(pd.read_csv(Path(folder)/"final_fold_results.csv")), 250)
            paired = pd.read_csv(Path(folder)/"paired_differences.csv")
            self.assertTrue(pd.isna(paired.iloc[0].macro_f1_difference))
            self.assertAlmostEqual(paired.iloc[1].macro_f1_difference, .01)


if __name__ == "__main__":
    unittest.main()
