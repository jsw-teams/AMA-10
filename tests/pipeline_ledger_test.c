#include "pipeline_ledger.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int contains(const char *path, const char *needle) {
    FILE *f = fopen(path, "rb");
    char buf[2048]; size_t n;
    if (!f) return 0;
    n = fread(buf, 1, sizeof(buf)-1, f); fclose(f); buf[n] = 0;
    return strstr(buf, needle) != NULL;
}

int main(void) {
    const char *path = "pipeline-ledger-test.jsonl";
    AMA10ShardEvent e = {
        "fineweb_edu", "sample/100BT/000_00004.parquet", "validated", "rev123",
        "https://example.invalid/a\"b", "data/cache/a\nb.parquet",
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        "data/processed/a.txt", "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
        100, 100, 123, 3, 42, 0
    };
    remove(path);
    if (!ama10_ledger_append_event(path, &e) ||
        !contains(path, "\\\"b") || !contains(path, "\\nb") ||
        !contains(path, "\"valid_tokens\":42") || !contains(path, "data/processed/a.txt") ||
        !contains(path, "\"processed_bytes\":123")) return 1;
    e.dataset_id = "unknown";
    if (ama10_ledger_append_event(path, &e)) return 1;
    e.dataset_id = "fineweb_edu"; e.validated_bytes = 101;
    if (ama10_ledger_append_event(path, &e)) return 1;
    e.validated_bytes = 100; e.consumed_tokens = 43;
    if (ama10_ledger_append_event(path, &e)) return 1;
    remove(path);
    puts("C11 shard ledger append, JSON escaping, and counter validation: ok");
    return 0;
}
