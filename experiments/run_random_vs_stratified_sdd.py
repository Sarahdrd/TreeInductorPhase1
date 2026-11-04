"""Spawn-safe entry point. Default mode is pilot; full/final require explicit modes."""
import os
import sys
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    from evolution.random_vs_stratified_sdd import main
    main()
