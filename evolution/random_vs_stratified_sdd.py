"""Isolated, resumable SDD experiment using the existing tree and GA operators.

Run via experiments/run_random_vs_stratified_sdd.py (Windows spawn safe).
No notebook execution, baseline fitting, or held-out monitoring occurs here.
"""
from __future__ import annotations

import os

# These must precede numeric imports, including in spawned interpreters.
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"

import argparse
import hashlib
import json
import multiprocessing as mp
import platform
import random
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from core.DatasetCharacterizer import (
    DatasetCharacterizer, TARGET_CONFIG, ID_COLUMNS, MISSING_TOKENS,
    NON_DUMMY_BASELINE_MODELS, minmax_normalize_on_pool,
    assign_predictive_difficulty_groups,
)
from experiments.characterization_persistence import data_digest, validate_evaluations

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "random_vs_stratified_sdd"
SOURCE = ROOT / "results" / "new_characterization_experiments"
GROUPS = ["easy", "intermediate", "hard"]
SELECTION_SEEDS = [7, 21, 42, 84, 123]
GA_SEEDS = [1007, 1021, 1042, 1084, 1123]
OUTER_SPLIT_SEED = 2026
PREPROCESSING = "fold-local sorted ordinal categories; unknown=NaN; preserve missing; no imputation v1"


def digest(value):
    raw = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def stable_seed(*parts):
    return int(digest(list(parts))[:8], 16)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def immutable_json(path, value):
    if Path(path).exists() and read_json(path) != value:
        raise ValueError(f"Incompatible existing artifact: {path}. Use a new output directory; never overwrite a protocol.")
    if not Path(path).exists():
        write_json(path, value)


@dataclass(frozen=True)
class Config:
    mu: int = 10
    lambd: int = 10
    generations: int = 10
    crossover_prob: float = 1.0
    mutation_prob: float = 0.1
    search_cv_folds: int = 5
    final_cv_folds: int = 5
    gen_dataset_timeout_s: float = 900.0
    n_workers: int = 4

    def __post_init__(self):
        if self.mu < 2 or self.lambd < 2 or self.lambd % 2:
            raise ValueError("mu >= 2 and an even lambda >= 2 are required")
        if self.search_cv_folds != 5 or self.final_cv_folds != 5:
            raise ValueError("Exactly five StratifiedKFold folds are required")
        if self.generations < 2 or self.n_workers < 1 or self.gen_dataset_timeout_s <= 0:
            raise ValueError("Invalid generation, worker or timeout configuration")
        if not 0 <= self.crossover_prob <= 1 or not 0 <= self.mutation_prob <= 1:
            raise ValueError("Probabilities must be in [0, 1]")


def sdd(dataset_means):
    """One definition for both search and reporting, reference=1, percentage scale."""
    values = np.asarray(dataset_means, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("SDD requires at least two complete finite dataset means")
    if ((values < 0) | (values > 1)).any():
        raise ValueError("Macro-F1 must be in [0, 1]")
    xi = 100 * np.abs(values - 1.0) / ((values + 1.0) / 2.0)
    return float(xi.std(ddof=1))


def aggregate(results, expected):
    complete = [name for name in expected if name in results and complete_result(results[name])]
    summary = dict(coverage=len(complete), expected_datasets=len(expected),
                   macro_f1=None, sdd=None, between_dataset_std=None)
    if len(complete) != len(expected) or len(expected) < 2:
        return summary
    means = [np.mean([f["macro_f1"] for f in results[name]["folds"]]) for name in expected]
    summary.update(macro_f1=float(np.mean(means)), sdd=sdd(means),
                   between_dataset_std=float(np.std(means, ddof=1)))
    return summary


def complete_result(result):
    folds = result.get("folds", [])
    return (result.get("status") == "ok" and len(folds) == 5
            and [f.get("fold") for f in folds] == list(range(5))
            and all(f.get("status") == "ok" and f.get("macro_f1") is not None
                    and np.isfinite(f["macro_f1"]) and 0 <= f["macro_f1"] <= 1 for f in folds))


def load_data(name, path):
    # Canonical loader removes missing targets before encoding and resolves IDs strictly.
    return DatasetCharacterizer()._load_dataset(name, Path(path))


def identity(name, path, obj, audit):
    return dict(path=str(Path(path).resolve()), file_sha256=digest(Path(path).read_bytes()),
                filtered_data_sha256=data_digest(obj), target=audit["target_col"],
                resolved_ids=[x for x in audit["dropped_id_cols"].split(", ") if x],
                n_samples=len(obj["y"]))


def partitions(y, name):
    if len(np.unique(y)) < 2 or np.bincount(y).min() < 5:
        raise ValueError(f"{name}: five observations per class required; no CV fallback")
    seed = stable_seed("partition", name, 2026)
    return seed, [[tr.tolist(), va.tolist()] for tr, va in
                  StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)]


def fold_features(train, validation):
    """The custom tree uses numeric thresholds, so encode categories within training only."""
    train, validation = train.copy(), validation.copy()
    for column in train:
        if not pd.api.types.is_numeric_dtype(train[column]):
            levels = sorted(train[column].dropna().astype(str).unique())
            mapping = {value: index for index, value in enumerate(levels)}
            train[column] = train[column].map(lambda x: mapping.get(str(x), np.nan) if pd.notna(x) else np.nan)
            validation[column] = validation[column].map(lambda x: mapping.get(str(x), np.nan) if pd.notna(x) else np.nan)
    return train.to_numpy(dtype=float), validation.to_numpy(dtype=float)


def rank_available(components, available):
    frame = components.set_index("dataset").loc[sorted(available)].reset_index()
    for field in ("E", "V"):
        frame["normalized_" + field] = minmax_normalize_on_pool(frame[field], pd.Series(True, index=frame.index))
    frame["difficulty_score"] = .70 * frame.normalized_E + .30 * frame.normalized_V
    grouping = frame.rename(columns={"F": "baseline_macro_f1_mean", "V": "cv_variability",
                                     "difficulty_score": "predictive_difficulty_score"}).assign(ranking_eligible=True)
    frame["difficulty_group"], frame["position"] = assign_predictive_difficulty_groups(grouping)
    return frame.sort_values("position").reset_index(drop=True)


def selections(available, ranking):
    rows = []
    for pair, (selection_seed, ga_seed) in enumerate(zip(SELECTION_SEEDS, GA_SEEDS), 1):
        # Independent local streams; random draws are never rejected.
        random_names = np.random.default_rng(selection_seed).choice(sorted(available), 3, replace=False).tolist()
        rng = np.random.default_rng(selection_seed)
        stratified_names = [str(rng.choice(sorted(ranking.loc[ranking.difficulty_group.eq(g), "dataset"]))) for g in GROUPS]
        for condition, names in [("random", random_names), ("stratified", stratified_names)]:
            rows.append(dict(run_id=f"pair_{pair:02d}_{condition}", pair=pair, condition=condition,
                             selection_seed=selection_seed, ga_seed=ga_seed, datasets=sorted(names)))
    return rows


def code_environment():
    files = sorted((ROOT / "core").glob("*.py")) + sorted((ROOT / "evolution").glob("*.py"))
    files += [ROOT / "experiments" / "characterization_persistence.py",
              ROOT / "experiments" / "run_random_vs_stratified_sdd.py"]
    return dict(code={str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in files},
                versions={p: version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "pymoo"]},
                python=platform.python_version(), system=platform.platform(), machine=platform.machine(),
                processor=platform.processor(), cpu_count=os.cpu_count(), numeric_threads=1)


def prepare(output=OUTPUT, source=SOURCE, config=Config()):
    """Validate historical individual results without fitting or executing any notebook."""
    output, source = Path(output), Path(source)
    metadata = read_json(source / "metadata.json")
    canonical_pool = read_json(source / "canonical_pool.json")
    protocol = metadata["protocol"]
    if (protocol["target_config"] != TARGET_CONFIG or protocol["id_columns"] != ID_COLUMNS
            or protocol["missing_tokens"] != MISSING_TOKENS
            or protocol["cv"] != dict(folds=5, repeats=3, seed=42) or protocol["std_ddof"] != 1):
        raise ValueError("Canonical target/ID/missing-token/CV protocol mismatch")
    datasets, rows, audits, exclusions = {}, [], [], []
    for name, path in sorted(DatasetCharacterizer.default_pool().items()):
        obj, audit = load_data(name, path)
        current = identity(name, path, obj, audit)
        audits.append(audit)
        if current != metadata["datasets"][name]:
            raise ValueError(f"Canonical data identity mismatch for {name}; required: {source / 'metadata.json'}")
        if audit["min_class_count"] < 5:
            exclusions.append(dict(dataset=name, reason=f"min_class_count={audit['min_class_count']} < 5"))
            continue
        individual = []
        for model in NON_DUMMY_BASELINE_MODELS:
            cache_path = source / "cache" / f"{name}__{model}.json"
            if not cache_path.exists():
                raise FileNotFoundError(f"Required canonical individual result missing: {cache_path}")
            payload = read_json(cache_path)
            spec = payload["specification"]
            if (spec["data"] != current or spec["protocol"] != protocol or spec["model"] != model
                    or payload["signature"] != digest(spec)):
                raise ValueError(f"Invalid canonical cache identity: {cache_path}")
            individual.append(payload["row"])
        frame = pd.DataFrame(individual)
        try:
            validate_evaluations(frame, [name], NON_DUMMY_BASELINE_MODELS)
        except ValueError as exc:
            exclusions.append(dict(dataset=name, reason=str(exc)))
            continue
        split_seed, split_list = partitions(obj["y"], name)
        datasets[name] = dict(name=name, identity=current, audit=audit, split_seed=split_seed,
                              splits=split_list, splits_sha256=digest(split_list))
        rows.extend(individual)
    eligible = sorted(datasets)
    if len(eligible) < 8:
        raise ValueError("At least eight eligible datasets are needed for five held out and three search strata")
    held_out = sorted(np.random.default_rng(OUTER_SPLIT_SEED).choice(eligible, 5, replace=False).tolist())
    available = sorted(set(eligible) - set(held_out))
    manifest = dict(outer_split_seed=OUTER_SPLIT_SEED, expected_size=17, observed_size=len(eligible),
                    eligible=eligible, held_out=held_out, available=available, exclusions=exclusions,
                    historical_eligible=sorted(canonical_pool["eligible"]),
                    discrepancy=len(eligible) != 17 or eligible != sorted(canonical_pool["eligible"]))
    # Reserve is persisted BEFORE ranking, selection, or any evolutionary evaluation.
    immutable_json(output / "reserve_manifest.json", manifest)
    baseline = pd.DataFrame(rows)
    search_baseline = baseline[baseline.dataset.isin(available)]
    components = search_baseline.groupby("dataset").agg(F=("f1_macro_mean", "mean"), V=("f1_macro_std", "mean")).reset_index()
    components["E"] = 1 - components.F
    ranking = rank_available(components, available)
    plan = selections(available, ranking)
    from evolution.genetic_DT_GA_Multi import generate_population, repair_gene, VAR_DOMAINS
    initial_populations = {}
    saved_random = random.getstate()
    try:
        for seed in GA_SEEDS:
            random.seed(seed)
            initial_populations[str(seed)] = [repair_gene(g) for g in generate_population(config.mu)]
    finally:
        random.setstate(saved_random)
    specification = dict(config=asdict(config), manifest=manifest, selections=plan,
                         initial_populations=initial_populations, domains=[list(d) for d in VAR_DOMAINS],
                         preprocessing=PREPROCESSING, environment=code_environment(),
                         target_config=TARGET_CONFIG, id_columns=ID_COLUMNS,
                         fitness="-std(100*abs(mean_fold_macro_f1-1)/((mean_fold_macro_f1+1)/2), ddof=1)",
                         baseline_source_sha256=digest(rows),
                         datasets=datasets, model_seed_rule="sha256(['model', dataset, gene, fold]) first uint32",
                         internal_fixed_seeds=dict(augmentation=42, pruning_validation_split=42),
                         tie_break="lexicographic gene, no additional objective")
    immutable_json(output / "protocol.json", specification)
    immutable_json(output / "selection_manifest.json", plan)
    immutable_json(output / "initial_populations.json", initial_populations)
    pd.DataFrame(audits).to_csv(output / "target_id_audit.csv", index=False)
    ranking.to_csv(output / "available_pool_ranking.csv", index=False)
    pd.DataFrame([{**r, "datasets": json.dumps(r["datasets"])} for r in plan]).to_csv(output / "selections.csv", index=False)
    pd.DataFrame([dict(dataset=d, role="held_out" if d in held_out else "available") for d in eligible]).to_csv(output / "reserve_manifest.csv", index=False)
    write_json(output / "characterization_cost.json", dict(
        reused=True, new_baseline_fits=0, source=str(source),
        historical_four_baseline_seconds=float(baseline.runtime_seconds.sum()),
        historical_available_pool_seconds=float(search_baseline.runtime_seconds.sum()),
        interpretation="Sum of recorded model evaluation runtimes, not current wall time; stratification requires this prior computation."))
    print(f"Canonical pool: {len(eligible)} (expected 17); held out: {held_out}; discrepancy: {manifest['discrepancy']}", flush=True)
    # Equality above validates the requested experiment, but Python considers 900
    # and 900.0 equal while their JSON digests differ. Continue with the exact
    # persisted representation so checkpoint and evaluation-cache identities stay
    # stable across CLI defaults, explicitly supplied flags and notebook launches.
    return read_json(output / "protocol.json")


def evaluation_spec(gene, dataset, protocol):
    return dict(gene=list(gene), dataset=dataset, preprocessing=protocol["preprocessing"],
                model_seeds=[stable_seed("model", dataset["name"], list(gene), fold) for fold in range(5)],
                internal_fixed_seeds=protocol["internal_fixed_seeds"],
                environment=protocol["environment"], timeout_s=protocol["config"]["gen_dataset_timeout_s"],
                scoring="macro_f1; all dataset labels; zero_division=0")


def _dataset_worker(connection, spec):
    """Report fold starts before fit so killed/incomplete fits remain countable."""
    started = time.perf_counter()
    folds = []
    try:
        from evolution.encode_decode import decode
        dataset = spec["dataset"]
        obj, audit = load_data(dataset["name"], dataset["identity"]["path"])
        if identity(dataset["name"], dataset["identity"]["path"], obj, audit) != dataset["identity"]:
            raise ValueError("Data changed after manifest creation")
        X, y = obj["X"], obj["y"]
        for fold, (train, valid) in enumerate(dataset["splits"]):
            start = time.perf_counter()
            # Import-time seeds in MissingValues have already run; reseed AFTER imports.
            seed = spec["model_seeds"][fold]
            random.seed(seed)
            np.random.seed(seed)
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                train_X, valid_X = fold_features(X.iloc[train], X.iloc[valid])
                tree = decode(spec["gene"])
                decode_seconds = time.perf_counter() - start
                connection.send(("fit_started", fold))
                fit_start = time.perf_counter()
                tree.fit(train_X, y[train])
                fit_seconds = time.perf_counter() - fit_start
                metric_start = time.perf_counter()
                score = float(f1_score(y[valid], tree.predict(valid_X), labels=np.unique(y), average="macro", zero_division=0))
                record = dict(fold=fold, status="ok", macro_f1=score, model_seed=seed,
                              decode_preprocess_seconds=decode_seconds, fit_seconds=fit_seconds,
                              metrics_seconds=time.perf_counter() - metric_start,
                              total_seconds=time.perf_counter() - start,
                              nodes=int(tree.count_nodes()), depth=int(tree.get_depth()),
                              warnings=sorted(set(str(w.message) for w in captured)))
            folds.append(record)
            connection.send(("fold", record))
        connection.send(("done", dict(status="ok", folds=folds, worker_seconds=time.perf_counter() - started)))
    except Exception:
        connection.send(("done", dict(status="error", error=traceback.format_exc(), folds=folds,
                                      worker_seconds=time.perf_counter() - started)))
    finally:
        connection.close()


def timed_dataset(spec):
    """Hard timeout per gene-dataset, including all five folds and process startup."""
    ctx = mp.get_context("spawn")
    receive, send = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_dataset_worker, args=(send, spec))
    started = time.perf_counter()
    result, folds, fits_started = None, [], 0
    try:
        process.start()
        send.close()
        deadline = started + spec["timeout_s"]
        while result is None:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                result = dict(status="timeout", error="gene-dataset hard timeout", folds=folds)
                break
            if receive.poll(min(.05, remaining)):
                try:
                    kind, value = receive.recv()
                except EOFError:
                    result = dict(status="error", error="worker exited without result", folds=folds)
                    break
                if kind == "fit_started":
                    fits_started += 1
                elif kind == "fold":
                    folds.append(value)
                elif kind == "done":
                    result = value
            elif not process.is_alive():
                # A final poll drains an already-sent result; never Queue.empty().
                if not receive.poll():
                    result = dict(status="error", error="worker exited without result", folds=folds)
    finally:
        if process.pid is not None:
            process.join(.2)
            if process.is_alive():
                process.terminate()
                process.join(5)
            if process.is_alive():
                process.kill()
                process.join()
            process.close()
        receive.close()
        send.close()
    result["wall_seconds"] = time.perf_counter() - started
    result["fits_started"] = fits_started
    result["fits_completed"] = len(result["folds"])
    result["process_overhead_seconds"] = (max(0., result["wall_seconds"] - result["worker_seconds"])
                                          if "worker_seconds" in result else None)
    return result


def _gene_worker(specs):
    # Same topology as the existing GA: genes in a process pool, timed dataset children.
    return [timed_dataset(spec) for spec in specs]


class Evaluator:
    def __init__(self, protocol, output, phase="search", use_cache=True):
        if phase not in ("search", "final"):
            raise ValueError("Unknown evaluation phase")
        self.protocol, self.output, self.phase, self.use_cache = protocol, Path(output), phase, use_cache
        self.cache = self.output / "cache"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.events = []

    def evaluate(self, genes, names):
        allowed = self.protocol["manifest"]["available" if self.phase == "search" else "held_out"]
        if not names or len(set(names)) != len(names) or not set(names) <= set(allowed):
            raise ValueError(f"Forbidden or duplicated datasets in {self.phase}: {names}")
        unique = list(dict.fromkeys(tuple(g) for g in genes))
        results = {gene: {} for gene in unique}
        jobs, references = [], []
        hits = 0
        for gene in unique:
            pending, refs = [], []
            for name in names:
                spec = evaluation_spec(gene, self.protocol["datasets"][name], self.protocol)
                key = digest(spec)
                cache_path = self.cache / f"{key}.json"
                payload = read_json(cache_path) if self.use_cache and cache_path.exists() else None
                if payload is not None:
                    if payload["specification"] != spec or not complete_result(payload["result"]):
                        raise ValueError(f"Invalid evaluation cache: {cache_path}")
                    result = payload["result"]
                    results[gene][name] = result
                    hits += 1
                    self.events.append(dict(gene=list(gene), dataset=name, cache_key=key, cache_hit=True,
                                            status="ok", wall_seconds=0., fits_started=0, fits_completed=0,
                                            process_overhead_seconds=0.))
                else:
                    pending.append(spec)
                    refs.append((gene, name, key, spec))
            if pending:
                jobs.append(pending)
                references.append(refs)
        if jobs:
            with ProcessPoolExecutor(max_workers=min(self.protocol["config"]["n_workers"], len(jobs)),
                                     mp_context=mp.get_context("spawn")) as executor:
                # map restores submission order, independent of process completion order.
                for refs, batch in zip(references, executor.map(_gene_worker, jobs)):
                    for (gene, name, key, spec), result in zip(refs, batch):
                        results[gene][name] = result
                        event = dict(gene=list(gene), dataset=name, cache_key=key, cache_hit=False,
                                     **{k: result[k] for k in ("status", "wall_seconds", "fits_started",
                                                             "fits_completed", "process_overhead_seconds")})
                        if result.get("error"):
                            event["error"] = result["error"]
                        self.events.append(event)
                        # Preserve failed partial folds for auditing, but never read them as cache.
                        attempt = self.output / "attempts" / f"{key}_{time.time_ns()}.json"
                        write_json(attempt, dict(specification=spec, result=result))
                        if self.use_cache and complete_result(result):
                            write_json(self.cache / f"{key}.json", dict(specification=spec, result=result))
        summaries = {gene: aggregate(results[gene], names) for gene in unique}
        fitness = [-summaries[tuple(g)]["sdd"] if summaries[tuple(g)]["sdd"] is not None else -np.inf for g in genes]
        return fitness, results, dict(candidates_requested=len(genes), batch_unique_genes=len(unique),
                                     duplicate_positions=len(genes) - len(unique), cache_hits=hits,
                                     dataset_requests=len(genes) * len(names), unique_dataset_requests=len(unique) * len(names))


def _tuple(value):
    return tuple(_tuple(v) for v in value) if isinstance(value, list) else value


def rng_state():
    state = np.random.get_state()
    return dict(python=random.getstate(), numpy=[state[0], state[1].tolist(), state[2], state[3], state[4]])


def restore_rng(state):
    random.setstate(_tuple(state["python"]))
    n = state["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))


def run_search(protocol, selection, output, target_generation, evaluator=None):
    """Compatible mu+lambda loop: existing generator/repair/crossover/tournament.

    Differences are explicit: deterministic gene ties, injected evaluator, persistence,
    and no unseen calls. The original generator's clamped initialization and crossover
    modulo behavior are intentionally preserved, not silently corrected.
    """
    from evolution.genetic_DT_GA_Multi import (
        generate_population, repair_gene, VAR_DOMAINS, tournament_selection, single_point_crossover,
    )
    output = Path(output)
    config = Config(**protocol["config"])
    if target_generation < 0 or target_generation > config.generations:
        raise ValueError("Requested generation exceeds the frozen budget")
    if selection not in protocol["selections"] or not set(selection["datasets"]) <= set(protocol["manifest"]["available"]):
        raise ValueError("Search selection is not in the prespecified manifest")
    evaluator = evaluator or Evaluator(protocol, output)
    folder = output / "runs" / selection["run_id"]
    checkpoint = folder / "checkpoint.json"
    signature = digest(dict(protocol=protocol, selection=selection))
    if checkpoint.exists():
        state = read_json(checkpoint)
        if state["signature"] != signature:
            raise ValueError(f"Checkpoint protocol changed: {checkpoint}")
        restore_rng(state["rng"])
        population = state["population"]
        fitness = [v if v is not None else -np.inf for v in state["fitness"]]
        evaluator.events = state["events"].copy()
        if state["status"] == "failed":
            return state
        first_generation = state["generation"] + 1
    else:
        random.seed(selection["ga_seed"])
        np.random.seed(selection["ga_seed"])
        population = [repair_gene(g) for g in generate_population(config.mu)]
        if population != protocol["initial_populations"][str(selection["ga_seed"])]:
            raise ValueError("Initial population differs from manifest")
        fitness = []
        state = dict(signature=signature, configuration=asdict(config), selection=selection,
                     cache_directory=str(evaluator.cache.resolve()), cache_protocol_id=digest(protocol),
                     history=[], events=[], seen_genes=[], initial_population=population)
        first_generation = 0
    for generation in range(first_generation, target_generation + 1):
        started = time.perf_counter()
        if generation == 0:
            proposed = population
        else:
            proposed = []
            for _ in range(config.lambd // 2):
                p1 = tournament_selection(population, fitness, k=2)
                p2 = tournament_selection(population, fitness, k=2)
                c1, c2 = single_point_crossover(p1, p2, VAR_DOMAINS, prob=config.crossover_prob)
                c1, c2 = repair_gene(c1), repair_gene(c2)
                # Preserve the original operator and random-number consumption order.
                c1 = [random.choice(list(VAR_DOMAINS[i])) if random.random() < config.mutation_prob else c1[i] for i in range(len(c1))]
                c2 = [random.choice(list(VAR_DOMAINS[i])) if random.random() < config.mutation_prob else c2[i] for i in range(len(c2))]
                proposed.extend([repair_gene(c1), repair_gene(c2)])
            assert len(proposed) == config.lambd
        event_start = len(evaluator.events)
        values, results, counts = evaluator.evaluate(proposed, selection["datasets"])
        seen = set(tuple(g) for g in state["seen_genes"])
        seen.update(tuple(g) for g in proposed)
        state["seen_genes"] = [list(g) for g in sorted(seen)]
        if generation == 0:
            fitness = values
        else:
            candidates, candidate_fitness = population + proposed, fitness + values
            indices = sorted(range(len(candidates)), key=lambda i: (-candidate_fitness[i], tuple(candidates[i]), i))[:config.mu]
            # Weak to strong order, as in the previous GA, with an explicit deterministic tie rule.
            indices = sorted(indices, key=lambda i: (candidate_fitness[i], tuple(candidates[i]), i))
            population, fitness = [candidates[i] for i in indices], [candidate_fitness[i] for i in indices]
        valid = [i for i, value in enumerate(fitness) if np.isfinite(value)]
        best = min(valid, key=lambda i: (-fitness[i], tuple(population[i]))) if valid else None
        events = evaluator.events[event_start:]
        state.update(generation=generation, population=population,
                     fitness=[float(v) if np.isfinite(v) else None for v in fitness],
                     best_gene=population[best] if best is not None else None,
                     best_fitness=float(fitness[best]) if best is not None else None,
                     status="failed" if best is None else ("complete" if generation == config.generations else "in_progress"))
        # Surviving parents may tie with other genes; recover their own metrics from prior populations.
        metrics = state.get("gene_metrics", {})
        metrics.update({json.dumps(list(g)): aggregate(r, selection["datasets"]) for g, r in results.items()})
        best_summary = metrics.get(json.dumps(state["best_gene"])) if best is not None else None
        state["gene_metrics"] = metrics
        state["history"].append(dict(generation=generation, seconds=time.perf_counter() - started,
            best_fitness=state["best_fitness"], best_gene=state["best_gene"], best_search_metrics=best_summary,
            cumulative_unique_genes=len(seen), fits_started=sum(e["fits_started"] for e in events),
            fits_completed=sum(e["fits_completed"] for e in events),
            timeouts=sum(e["status"] == "timeout" for e in events), errors=sum(e["status"] == "error" for e in events), **counts))
        state["events"] = evaluator.events.copy()
        state["rng"] = rng_state()
        write_json(checkpoint, state)
        pd.DataFrame(state["history"]).to_csv(folder / "history.csv", index=False)
        pd.DataFrame(state["events"]).to_csv(folder / "evaluation_events.csv", index=False)
        print(f"{selection['run_id']} generation {generation}: fitness={state['best_fitness']} "
              f"seconds={state['history'][-1]['seconds']:.1f} fits={state['history'][-1]['fits_started']} "
              f"status={state['status']}", flush=True)
        if state["status"] == "failed":
            break
    if state["status"] == "complete":
        immutable_json(folder / "winner.json", dict(selection=selection, gene=state["best_gene"],
                       fitness=state["best_fitness"], search_metrics=state["gene_metrics"][json.dumps(state["best_gene"])],
                       protocol_id=digest(protocol), generation=state["generation"]))
    return state


def pilot_report(states, output):
    rows, dataset_rows = [], []
    for state in states:
        history = state["history"][:3]
        events = state["events"]
        # A pilot report is written at pilot time; never infer a pilot from later full-run events.
        initial = history[0]["seconds"]
        generation_times = [r["seconds"] for r in history[1:]]
        estimate = initial + 10 * np.mean(generation_times) if len(generation_times) == 2 else None
        row = dict(**{k: state["selection"][k] for k in ("run_id", "condition", "pair")},
            status=state["status"], generations_completed=state["generation"],
            measured_search_seconds=sum(r["seconds"] for r in history), initial_seconds=initial,
            generation_1_seconds=generation_times[0] if generation_times else None,
            generation_2_seconds=generation_times[1] if len(generation_times) == 2 else None,
            candidates_requested=sum(r["candidates_requested"] for r in history),
            unique_genes=history[-1]["cumulative_unique_genes"],
            cache_hits=sum(r["cache_hits"] for r in history),
            fits_started=sum(r["fits_started"] for r in history),
            fits_completed=sum(r["fits_completed"] for r in history),
            timeouts=sum(r["timeouts"] for r in history), errors=sum(r["errors"] for r in history),
            estimated_10_generation_seconds=float(estimate) if estimate is not None else None,
            measured_process_overhead_task_seconds=sum(e["process_overhead_seconds"] or 0 for e in events),
            overhead_tasks_measurable=sum(e["process_overhead_seconds"] is not None and not e["cache_hit"] for e in events))
        rows.append(row)
        for name in state["selection"]["datasets"]:
            selected = [e for e in events if e["dataset"] == name]
            dataset_rows.append(dict(run_id=row["run_id"], condition=row["condition"], dataset=name,
                measured_task_seconds=sum(e["wall_seconds"] for e in selected),
                fits_started=sum(e["fits_started"] for e in selected),
                cache_hits=sum(e["cache_hit"] for e in selected),
                timeouts=sum(e["status"] == "timeout" for e in selected),
                errors=sum(e["status"] == "error" for e in selected)))
    pd.DataFrame(rows).to_csv(Path(output) / "pilot_summary.csv", index=False)
    pd.DataFrame(dataset_rows).to_csv(Path(output) / "pilot_dataset_cost.csv", index=False)
    estimates = [r["estimated_10_generation_seconds"] for r in rows]
    write_json(Path(output) / "pilot_cost_estimate.json", dict(
        measured_conditions=rows,
        extrapolated_full_plan_seconds=5 * sum(estimates) if len(estimates) == 2 and all(x is not None for x in estimates) else None,
        excludes="characterization, setup, final evaluation, and report rendering",
        caveat="Rough extrapolation from one pair. Genes, datasets, failures and duplicate/cache frequency change cost. Task seconds overlap across workers; overhead is measurable only for workers returning a result. No speedup claim."))
    return pd.DataFrame(rows)


def freeze_winners(protocol, output):
    output = Path(output)
    winners = []
    for selection in protocol["selections"]:
        folder = output / "runs" / selection["run_id"]
        if not (folder / "winner.json").exists():
            raise ValueError(f"All ten searches must complete before final evaluation; missing {folder / 'winner.json'}")
        winner, state = read_json(folder / "winner.json"), read_json(folder / "checkpoint.json")
        if (state["status"] != "complete" or state["generation"] != protocol["config"]["generations"]
                or winner["protocol_id"] != digest(protocol) or winner["selection"] != selection
                or state["signature"] != digest(dict(protocol=protocol, selection=selection))
                or winner["gene"] != state["best_gene"] or winner["fitness"] != state["best_fitness"]
                or winner["fitness"] is None or not np.isfinite(winner["fitness"])):
            raise ValueError(f"Invalid or incompatible winner: {folder}")
        expected_budget = protocol["config"]["mu"] + protocol["config"]["lambd"] * protocol["config"]["generations"]
        if sum(h["candidates_requested"] for h in state["history"]) != expected_budget:
            raise ValueError("Logical candidate budget mismatch")
        winners.append(winner)
    immutable_json(output / "frozen_winners.json", dict(protocol_id=digest(protocol), winners=winners))
    pd.DataFrame([{**w["selection"], "gene": json.dumps(w["gene"]), "search_fitness": w["fitness"],
                   "search_macro_f1": w["search_metrics"]["macro_f1"]} for w in winners]).to_csv(output / "winners.csv", index=False)
    return winners


def final_evaluation(protocol, output):
    output = Path(output)
    # Freeze and validate ALL winners before creating a final evaluator or accessing reserved data.
    winners = freeze_winners(protocol, output)
    evaluator = Evaluator(protocol, output, phase="final")
    names = protocol["manifest"]["held_out"]
    summaries, dataset_rows, fold_rows = [], [], []
    for winner in winners:
        selection, gene = winner["selection"], winner["gene"]
        path = output / "final" / f"{selection['run_id']}.json"
        signature = digest(dict(winner=winner, protocol=protocol))
        if path.exists():
            saved = read_json(path)
            if saved["signature"] != signature:
                raise ValueError(f"Incompatible final result: {path}")
            results = saved["results"]
        else:
            start = time.perf_counter()
            _, evaluated, counts = evaluator.evaluate([gene], names)
            results = evaluated[tuple(gene)]
            write_json(path, dict(signature=signature, results=results, counts=counts,
                                 wall_seconds=time.perf_counter() - start))
        common = {k: selection[k] for k in ("run_id", "condition", "pair")}
        summary = aggregate(results, names)
        summaries.append(dict(**common, **summary, status="ok" if summary["macro_f1"] is not None else "incomplete",
                              failures=sum(r["status"] != "ok" for r in results.values()),
                              evaluation_task_seconds=sum(r["wall_seconds"] for r in results.values())))
        for name in names:
            result = results[name]
            complete = complete_result(result)
            dataset_rows.append(dict(**common, dataset=name, status=result["status"],
                macro_f1=float(np.mean([f["macro_f1"] for f in result["folds"]])) if complete else None,
                completed_folds=len(result["folds"]), task_seconds=result["wall_seconds"], error=result.get("error", "")))
            by_fold = {f["fold"]: f for f in result["folds"]}
            for fold in range(5):
                record = by_fold.get(fold, dict(fold=fold, status=result["status"], macro_f1=None,
                    model_seed=stable_seed("model", name, gene, fold), error=result.get("error", "not completed")))
                fold_rows.append(dict(**common, dataset=name, **record))
        print(f"Final {selection['run_id']}: coverage={summary['coverage']}/5 macro-F1={summary['macro_f1']}", flush=True)
    pd.DataFrame(fold_rows).to_csv(output / "final_fold_results.csv", index=False)
    pd.DataFrame(dataset_rows).to_csv(output / "final_dataset_results.csv", index=False)
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(output / "final_run_summary.csv", index=False)
    paired = summary_frame[summary_frame.condition.eq("random")].merge(
        summary_frame[summary_frame.condition.eq("stratified")], on="pair", suffixes=("_random", "_stratified"), validate="one_to_one")
    for metric in ("macro_f1", "sdd", "between_dataset_std"):
        paired[metric + "_difference"] = paired[metric + "_stratified"] - paired[metric + "_random"]
    paired.to_csv(output / "paired_differences.csv", index=False)
    return summary_frame


def figures(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    destination = output / "figures"
    destination.mkdir(exist_ok=True)

    def save(fig, name):
        fig.tight_layout()
        for extension in ("png", "pdf"):
            fig.savefig(destination / f"{name}.{extension}", dpi=180, bbox_inches="tight")
        plt.close(fig)

    histories = []
    for file in sorted((output / "runs").glob("*/checkpoint.json")):
        state = read_json(file)
        histories.extend([{**h, **{k: state["selection"][k] for k in ("run_id", "condition", "pair")}} for h in state["history"]])
    if histories:
        frame = pd.DataFrame(histories)
        frame.to_csv(output / "search_histories.csv", index=False)
        fig, ax = plt.subplots(figsize=(9, 5))
        for name, group in frame.groupby("run_id", sort=True):
            ax.plot(group.generation, group.best_fitness, marker="o", label=name)
        ax.set(xlabel="Completed generation (0 = initial population)", ylabel="Search fitness (-SDD, percentage scale)",
               title="Search diagnostic: fitness on different selected datasets")
        ax.legend(fontsize=7)
        save(fig, "search_fitness_diagnostic")
    if not (output / "final_run_summary.csv").exists():
        return  # Never fabricate final results after a pilot.
    summary = pd.read_csv(output / "final_run_summary.csv")
    for metric, label in [("macro_f1", "Mean held-out dataset macro-F1"), ("sdd", "Held-out SDD (percentage scale)")]:
        fig, ax = plt.subplots(figsize=(7, 5))
        for pair, group in summary.groupby("pair"):
            group = group.set_index("condition").reindex(["random", "stratified"])
            ax.plot([0, 1], group[metric], marker="o", label=f"Pair {pair}")
        ax.set(xticks=[0, 1], xticklabels=["random", "stratified"], ylabel=label,
               title="Fixed external reserve; paired descriptive results")
        ax.legend()
        save(fig, "final_" + metric)
    paired = pd.read_csv(output / "paired_differences.csv")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(paired.pair, paired.macro_f1_difference)
    ax.axhline(0, color="black", linewidth=1)
    ax.set(xlabel="Pair", ylabel="Macro-F1: stratified minus random", title="Paired differences (unavailable pairs remain missing)")
    save(fig, "paired_macro_f1_differences")
    data = pd.read_csv(output / "final_dataset_results.csv")
    fig, axes = plt.subplots(1, 5, figsize=(16, 4), sharey=True)
    for ax, (name, group) in zip(axes, data.groupby("dataset", sort=True)):
        for pair, pair_data in group.groupby("pair"):
            pair_data = pair_data.set_index("condition").reindex(["random", "stratified"])
            ax.plot([0, 1], pair_data.macro_f1, marker="o", alpha=.7)
        ax.set(title=name, xticks=[0, 1], xticklabels=["random", "stratified"], ylim=(0, 1))
        ax.tick_params(axis="x", rotation=35)
    axes[0].set_ylabel("Dataset mean macro-F1")
    save(fig, "held_out_dataset_results")


def execute(mode="pilot", output=OUTPUT, config=Config(), source=SOURCE):
    protocol = prepare(output, source, config)
    if mode == "smoke":
        # Infrastructure smoke uses one available dataset and a fixed gene; no scientific claims.
        from evolution.genetic_DT_GA_Multi import repair_gene
        name = sorted(protocol["manifest"]["available"])[0]
        gene = repair_gene([0, 0, 20, 0, 0, 0, 0, 1])
        evaluator = Evaluator(protocol, output)
        _, results, counts = evaluator.evaluate([gene, gene], [name])
        write_json(Path(output) / "smoke_results.json", dict(dataset=name, gene=gene,
                   counts=counts, result=results[tuple(gene)][name]))
        return
    if mode == "pilot":
        states = []
        for selection in protocol["selections"][:2]:
            existing = Path(output) / "runs" / selection["run_id"] / "checkpoint.json"
            if existing.exists() and read_json(existing)["generation"] > 2:
                raise ValueError("Pilot already continued beyond generation 2; use full_search")
            states.append(run_search(protocol, selection, output, target_generation=2))
        print(pilot_report(states, output).to_string(index=False), flush=True)
    elif mode == "full_search":
        for selection in protocol["selections"]:
            run_search(protocol, selection, output, target_generation=config.generations)
        freeze_winners(protocol, output)
    elif mode == "final_evaluation":
        final_evaluation(protocol, output)
    else:
        raise ValueError(f"Unknown execution mode: {mode}")
    figures(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "pilot", "full_search", "final_evaluation"], default="pilot")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--n-workers", type=int, default=4)
    parser.add_argument("--gen-dataset-timeout-s", type=float, default=900)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    config = Config(n_workers=args.n_workers, gen_dataset_timeout_s=args.gen_dataset_timeout_s)
    if args.prepare_only:
        prepare(args.output, args.source, config)
    else:
        execute(args.mode, args.output, config, args.source)
