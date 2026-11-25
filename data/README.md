# Dataset availability and exact identities

The original locally curated CSVs are intentionally not uploaded automatically.
`experiments/configs/dataset_manifest.json` lists all 20 descriptive datasets,
the 17 eligible tasks, exact filenames, SHA-256 values, parsed-data fingerprints,
target columns, class counts and row-order rules. Obtain the exact curated files
from the study archive/maintainers, place them at the manifest's relative paths,
and verify their hashes before using the saved partitions. Original targets,
including five Heart Disease classes and six nominal Wine Quality Red classes,
must be retained. A generic upstream download is not guaranteed to have the same
rows, encoding or target definition. Upstream identifiers for these curated
versions are not verified; no substitute download instructions are invented.

MAGIC Gamma Telescope is separately available from
[UCI](https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope), DOI
10.24432/C52C8B. The evaluated local `magic04.data` has 19,020 rows and SHA-256
`e9314b7ebd4b4b59a3b3d65f7316663963777b16a46786877651dbbaa640b36a`.
It is comma-separated and is parsed directly without CSV conversion. Verify the
hash; a remote byte-for-byte download was not asserted during this cleanup.
MAGIC is not one of the 17 ranked datasets and has no difficulty stratum.

Exact full reproduction additionally needs the local historical `results/`
archive, all official record stores, prepared MAGIC subsample/CV manifests and
indices, and the recorded external-solver sources/runtimes. These remain local;
no public archive accession has yet been verified. The distributed final tables
and figures can be inspected without these private archive dependencies.
