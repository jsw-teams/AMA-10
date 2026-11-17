#ifndef AMA10_PARQUET_INGEST_H
#define AMA10_PARQUET_INGEST_H

int ama10_parquet_ingest(const char *raw_path, const char *processed_path,
                         const char *ledger_path, const char *dataset_id,
                         const char *shard_id, const char *source_revision,
                         const char *source_url, const char *expected_sha256,
                         double minimum_quality_score);

#endif
