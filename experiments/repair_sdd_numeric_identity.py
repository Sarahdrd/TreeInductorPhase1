"""Audited one-time recovery of the original Ex4 numeric-identity bug.

Default is a dry run. --apply preserves the existing scientific state and creates
new cache identities only after proving that the sole executable source change
is prepare() returning the already-validated persisted protocol. No fits occur.
Original caches and attempts remain unchanged; the supplied backup is mandatory.
"""
import argparse
import ast
import copy
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evolution import random_vs_stratified_sdd as ex

ORIGINAL_SOURCE_SHA256 = "979941c0835216b83f4305942709b1de5da126cfec1df8d0f3edd5319caf617c"
MODULE = Path("evolution/random_vs_stratified_sdd.py")


def verify_source_change(original, current):
    if ex.digest(original) != ORIGINAL_SOURCE_SHA256:
        raise ValueError("Not the reviewed original source version")
    tree = ast.parse(original.decode("utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "prepare")
    if ast.dump(function.body[-1]) != ast.dump(ast.parse("return specification").body[0]):
        raise ValueError("Unexpected original prepare() return")
    function.body[-1] = ast.parse('return read_json(output / "protocol.json")').body[0]
    if ast.dump(tree) != ast.dump(ast.parse(current.decode("utf-8"))):
        raise ValueError("Other executable code changed; recovery is not authorized by this migration")


def recover(output, backup, apply=False):
    output, backup = Path(output), Path(backup)
    old = ex.read_json(backup / "results/random_vs_stratified_sdd/protocol.json")
    old_source = (backup / MODULE).read_bytes()
    current_source = (ex.ROOT / MODULE).read_bytes()
    verify_source_change(old_source, current_source)
    key = str(MODULE)
    if old["environment"]["code"][key] != ORIGINAL_SOURCE_SHA256:
        raise ValueError("Backup protocol is not bound to the original source")
    expected = copy.deepcopy(old)
    expected["environment"]["code"][key] = ex.digest(current_source)
    # Verify current data, baseline inputs, every other code file and environment.
    with tempfile.TemporaryDirectory() as folder:
        actual = ex.prepare(folder, config=ex.Config(**old["config"]))
    if actual != expected or ex.digest(actual) != ex.digest(expected):
        raise ValueError("Scientific protocol, data or environment also changed; refuse migration")
    if ex.read_json(output / "protocol.json") != old:
        raise ValueError("Live protocol differs from backup; inspect before proceeding")

    checkpoints = []
    for path in sorted((output / "runs").glob("*/checkpoint.json")):
        state = ex.read_json(path)
        selection = state["selection"]
        if (selection not in old["selections"] or state["cache_protocol_id"] != ex.digest(old)
                or state["signature"] != ex.digest(dict(protocol=old, selection=selection))):
            raise ValueError(f"Checkpoint cannot be authenticated: {path}")
        original_path = backup / path.relative_to(ex.ROOT)
        if not original_path.exists() or original_path.read_bytes() != path.read_bytes():
            raise ValueError(f"Missing or stale checkpoint backup: {original_path}")
        replacement = copy.deepcopy(state)
        replacement["signature"] = ex.digest(dict(protocol=expected, selection=selection))
        replacement["cache_protocol_id"] = ex.digest(expected)
        checkpoints.append((path, replacement))

    cache_entries, mappings = [], []
    for path in sorted((output / "cache").glob("*.json")):
        payload = ex.read_json(path)
        spec = payload["specification"]
        if (ex.digest(spec) != path.stem or spec["environment"] != old["environment"]
                or not ex.complete_result(payload["result"])):
            raise ValueError(f"Cache entry cannot be authenticated: {path}")
        replacement = copy.deepcopy(payload)
        replacement["specification"]["environment"] = expected["environment"]
        new_key = ex.digest(replacement["specification"])
        target = path.with_name(new_key + ".json")
        if target.exists() and ex.read_json(target) != replacement:
            raise ValueError(f"Conflicting migrated cache: {target}")
        cache_entries.append((target, replacement))
        mappings.append(dict(original=path.stem, migrated=new_key))

    report = dict(reason="JSON numeric representation mismatch: 900 versus 900.0",
        proof="Entire module AST unchanged except prepare() returning validated protocol.json; scientific configuration, data, environment, operators, evaluator and RNG unchanged.",
        old_source_sha256=ex.digest(old_source), new_source_sha256=ex.digest(current_source),
        old_protocol_id=ex.digest(old), new_protocol_id=ex.digest(expected),
        checkpoint_count=len(checkpoints), reusable_complete_cache_entries=len(cache_entries),
        original_cache_and_attempts_preserved=True, cache_key_mapping=mappings,
        scores_fitness_population_rng_and_history_unchanged=True, model_fits=0,
        backup=str(backup.resolve()), applied=apply)
    if apply:
        # Originals remain available. Commit the protocol only after the new cache
        # files and checkpoint identities have been written successfully.
        for path, payload in cache_entries:
            ex.immutable_json(path, payload)
        for path, state in checkpoints:
            ex.write_json(path, state)
        ex.write_json(output / "protocol.json", expected)
        ex.write_json(backup / "migration_report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "cache_key_mapping"}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ex.OUTPUT)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    recover(args.output, args.backup, args.apply)
