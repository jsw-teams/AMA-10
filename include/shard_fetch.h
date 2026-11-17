#ifndef AMA10_SHARD_FETCH_H
#define AMA10_SHARD_FETCH_H

#include <stdint.h>

typedef struct {
    const char *ledger_path;
    const char *dataset_id;
    const char *shard_id;
    const char *source_revision;
    const char *url;
    const char *output_path;
    const char *expected_sha256;
    uint64_t max_bytes;
    unsigned minimum_delay_seconds;
} AMA10FetchRequest;

int ama10_fetch_official_shard(const AMA10FetchRequest *request, char sha256_hex[65], uint64_t *downloaded_bytes);

#endif
