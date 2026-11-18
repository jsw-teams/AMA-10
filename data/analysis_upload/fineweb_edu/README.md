# FineWeb-Edu cleaned shard for analysis

The cleaned training shard is split into Git LFS compatible byte chunks. The validation split is stored intact. Reassemble the training file and verify all SHA-256 hashes with `scripts/assemble_fineweb_upload.ps1`.

- Training bytes: 3,615,993,637
- Validation bytes: 188,435,943
- Training chunks: `train-00001-of-00013-train.part001`, `train-00001-of-00013-train.part002`, `train-00001-of-00013-train.part003`
- Training SHA-256: `5345fe1ffa08c747e44c5a119c6074dffa169c2ee9896088d9e9ab59d95a250c`
- Validation SHA-256: `09d22ac3db549c44068ac033e8a20770c9bf83262d32178a47952a892bd9673b`
