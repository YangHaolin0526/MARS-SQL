# Prepared datasets

This directory stores the prepared Parquet inputs consumed by the training and
inference entry points. The SQLite databases themselves are distributed by the
benchmark owners and are not included here; pass their root directory through
`DB_PATH`.

| File | Role indicated by the artifact name | SHA-256 |
| :--- | :--- | :--- |
| `bird_train.parquet` | BIRD training input | `e73406708080fd1a87174f1ea700a20ae53dab21f8ef0ab5306dfd3a8830e52d` |
| `bird_test.parquet` | BIRD inference/evaluation input | `f2d4cb49ece22199ee8b87be75937c27c93b58411dbb71ef16c97cec393bd410` |
| `validation.parquet` | Training validation input | `ece79bd0f0b99a039aab2e5fd21ff0bac21c590a159f13587ba2d1d8d0577117` |
| `spider_test.parquet` | Spider evaluation input | `6853362f6575e4985add7fea48e7747140fdfa5ff155292e1fbb7911ce0c1b6b` |
| `spider-DK_test.parquet` | Spider-DK evaluation input | `90b50a4c2abbabd6467e68ae565a33d2a1e336557a3f8f9bc8385e452cca568a` |

Verify downloaded or copied artifacts from the repository root with:

```bash
shasum -a 256 data/*.parquet
```

If an artifact is intentionally regenerated, update its checksum in this file in
the same commit so experiments can identify the exact input version.
