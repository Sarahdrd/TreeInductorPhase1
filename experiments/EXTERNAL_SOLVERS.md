# External solver setup and provenance

Reviewing exported results requires neither solver. Full execution was validated
on Windows with the v2 Windows job-object controller, serial execution, one
numerical thread, 120 seconds per process and 4 GiB job committed memory.

The unchanged bridges are in `block8-optimal-trees-v1/adapters/`. Author sources
and sidecars are local and ignored. Obtain the author repositories at these exact
commits (do not replace them with a current branch):

| Component | Source / version | Expected local location |
|---|---|---|
| MurTree | github.com/MurTree/murtree @ 129e0466acf43f5b4759fba8222800ce40aa32a3 | block8-optimal-trees-v1/sources/murtree |
| Quant-BnB | github.com/mengxianglgal/Quant-BnB @ 0a381a5bc6d689f66e5fca34a767da2c21918da0 | block8-optimal-trees-v1/sources/Quant-BnB |
| Compiler | portable Zig 0.13.0, C++14 | block8-optimal-trees-v1/sidecars/ |
| Julia | portable 1.10.10 | block8-optimal-trees-v1/sidecars/julia-1.10.10/ |
| Julia project | JSON 0.21.4 and locked dependencies | block8-optimal-trees-v1/sidecars/julia-project/ |

Restore the archived Julia Project.toml/Manifest.toml from the feasibility
analysis directory into the expected project location and instantiate it using
that Julia executable with a local JULIA_DEPOT_PATH. No global Julia or compiler
installation is necessary. The download manifest records archive identities and
official checksum verification. Build the MurTree bridge with a C++14 compiler
using the author implementation's translation units; the archived build commands
and logs identify the exact invocation. A binary built by a different compiler
may differ in checksum and requires a separately versioned execution manifest.

The immutable official configuration pins source, bridges, executable, Julia
project, Python environment and saved CV hashes. The final scientific export's
`configuration_summary.json` is a portable summary, **not a replacement for that
identity**. `block8_full.py` deliberately rejects changed provenance. For exact
archive verification, restore the original build and runtime artifacts. For a
new execution on another platform/location, prepare a separately versioned
configuration rather than editing old records or disabling checks. An automated
fresh-machine solver bootstrap has not been validated; reproducing exact binary
hashes from a fresh compiler installation is not claimed.

MurTree's MIT source is obtained from its authors. No explicit license file was
found in the pinned Quant-BnB source; it is not vendored here. Confirm the authors'
terms before redistribution. No dependency installation or new solver execution
was performed during final repository preparation.
