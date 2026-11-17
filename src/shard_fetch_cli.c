#include "shard_fetch.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

static int number(const char *s, uint64_t *value) {
    char *end = NULL; unsigned long long n;
    if (!s || !*s || *s == '-') return 0;
    errno = 0; n = strtoull(s, &end, 10);
    if (errno || !end || *end) return 0;
    *value = (uint64_t)n; return 1;
}

int main(int argc, char **argv) {
    AMA10FetchRequest r; uint64_t delay, bytes = 0; char sha[65];
    if (argc != 10) {
        fprintf(stderr, "Usage: %s <ledger.jsonl> <dataset_id> <shard_id> <commit-or-release> <https-url> <output_path> <expected_sha256|-> <max_bytes> <minimum_delay_seconds>\n", argv[0]);
        return 2;
    }
    if (!number(argv[8], &r.max_bytes) || !number(argv[9], &delay) || delay < 30 || delay > 3600) {
        fputs("max_bytes must be positive; minimum_delay_seconds must be 30..3600.\n", stderr); return 2;
    }
    r.ledger_path=argv[1]; r.dataset_id=argv[2]; r.shard_id=argv[3]; r.source_revision=argv[4];
    r.url=argv[5]; r.output_path=argv[6]; r.expected_sha256=argv[7]; r.minimum_delay_seconds=(unsigned)delay;
    if (!ama10_fetch_official_shard(&r, sha, &bytes)) { fputs("Shard fetch failed or did not validate. Partial remains resumable when a server validator was received.\n", stderr); return 1; }
    printf("downloaded_bytes=%llu sha256=%s output=%s\n", (unsigned long long)bytes, sha, r.output_path);
    return 0;
}
