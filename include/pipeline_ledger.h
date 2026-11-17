#ifndef AMA10_PIPELINE_LEDGER_H
#define AMA10_PIPELINE_LEDGER_H

#include <stdint.h>

typedef struct {
    const char *dataset_id;
    const char *shard_id;
    const char *state;
    const char *source_revision;
    const char *source_url;
    const char *local_path;
    const char *sha256;
    const char *processed_path;
    const char *processed_sha256;
    uint64_t acquired_bytes;
    uint64_t validated_bytes;
    uint64_t processed_bytes;
    uint64_t documents;
    uint64_t valid_tokens;
    uint64_t consumed_tokens;
} AMA10ShardEvent;

int ama10_ledger_append_event(const char *ledger_path, const AMA10ShardEvent *event);

#endif
