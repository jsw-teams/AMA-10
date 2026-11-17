#include "pipeline_ledger.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

static int number(const char *s, uint64_t *v) {
    char *end = NULL;
    unsigned long long n;
    if (!s || !*s || *s == '-') return 0;
    errno = 0;
    n = strtoull(s, &end, 10);
    if (errno || !end || *end) return 0;
    *v = (uint64_t)n;
    return 1;
}

int main(int argc, char **argv) {
    AMA10ShardEvent e;
    if (argc != 14 && argc != 17) {
        fprintf(stderr, "Usage: %s <ledger.jsonl> <dataset_id> <shard_id> <state> <revision> <url> <local_path> <sha256|-> <acquired_bytes> <validated_bytes> <documents> <valid_tokens> <consumed_tokens> [processed_path processed_sha256 processed_bytes]\n", argv[0]);
        return 2;
    }
    e.dataset_id = argv[2]; e.shard_id = argv[3]; e.state = argv[4];
    e.source_revision = argv[5]; e.source_url = argv[6]; e.local_path = argv[7]; e.sha256 = argv[8];
    e.processed_path = NULL; e.processed_sha256 = NULL; e.processed_bytes = 0;
    if (argc == 17) {
        e.processed_path = argv[14]; e.processed_sha256 = argv[15];
        if (!number(argv[16], &e.processed_bytes)) {
            fputs("processed_bytes must be a non-negative decimal integer.\n", stderr);
            return 2;
        }
    }
    if (!number(argv[9], &e.acquired_bytes) || !number(argv[10], &e.validated_bytes) ||
        !number(argv[11], &e.documents) || !number(argv[12], &e.valid_tokens) ||
        !number(argv[13], &e.consumed_tokens)) {
        fputs("Counters must be non-negative decimal integers.\n", stderr);
        return 2;
    }
    if (!ama10_ledger_append_event(argv[1], &e)) {
        fputs("Could not append shard event; verify dataset/state, counters, SHA-256, and ledger path.\n", stderr);
        return 1;
    }
    return 0;
}
