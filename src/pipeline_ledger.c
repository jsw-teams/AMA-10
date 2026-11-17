#include "pipeline_ledger.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#endif

static const char *const dataset_ids[] = {
    "fineweb_edu", "dclm_baseline", "cci3_hq", "skypile_150b", "wanjuan_1_0",
    "wanjuan_2_0", "fineweb_2", "culturax", "hplt_2", "the_stack_v2",
    "openwebmath", "finemath", "datacomp_1b", "dfn_5b", "wukong", "zero",
    "sharegpt4v", "allava", "synthdog", "docvqa", "textvqa", "renderedtext",
    "cambrian_10m", "the_cauldron", "llava_onevision", "fleurs",
    "bdd100k", "nuscenes", "waymo_open_dataset", "argoverse_2",
    "kitti", "a2d2", "comma2k19"
};
static const char *const states[] = {
    "planned", "downloading", "downloaded", "validated", "processed",
    "tokenized", "consumed", "failed_retryable", "failed_terminal"
};

static int member(const char *s, const char *const *items, size_t count) {
    size_t i;
    if (!s || !*s) return 0;
    for (i = 0; i < count; ++i) if (strcmp(s, items[i]) == 0) return 1;
    return 0;
}

static int valid_hash(const char *s) {
    size_t i;
    if (!s || strcmp(s, "-") == 0) return 1;
    if (strlen(s) != 64) return 0;
    for (i = 0; i < 64; ++i) if (!isxdigit((unsigned char)s[i])) return 0;
    return 1;
}

static size_t escaped_size(const char *s) {
    const unsigned char *p = (const unsigned char *)s;
    size_t n = 0;
    while (*p) {
        if (*p == '"' || *p == '\\' || *p == '\b' || *p == '\f' ||
            *p == '\n' || *p == '\r' || *p == '\t') n += 2;
        else if (*p < 0x20) n += 6;
        else n++;
        ++p;
    }
    return n;
}

static char *append_escaped(char *out, const char *s) {
    const unsigned char *p = (const unsigned char *)s;
    static const char hex[] = "0123456789abcdef";
    while (*p) {
        switch (*p) {
            case '"': *out++ = '\\'; *out++ = '"'; break;
            case '\\': *out++ = '\\'; *out++ = '\\'; break;
            case '\b': *out++ = '\\'; *out++ = 'b'; break;
            case '\f': *out++ = '\\'; *out++ = 'f'; break;
            case '\n': *out++ = '\\'; *out++ = 'n'; break;
            case '\r': *out++ = '\\'; *out++ = 'r'; break;
            case '\t': *out++ = '\\'; *out++ = 't'; break;
            default:
                if (*p < 0x20) {
                    *out++ = '\\'; *out++ = 'u'; *out++ = '0'; *out++ = '0';
                    *out++ = hex[*p >> 4]; *out++ = hex[*p & 15];
                } else *out++ = (char)*p;
        }
        ++p;
    }
    return out;
}

static int write_all(const char *path, const char *buf, size_t n) {
#ifdef _WIN32
    HANDLE h;
    OVERLAPPED ov;
    DWORD locked = 0;
    size_t done = 0;
    int ok = 0;
    h = CreateFileA(path, FILE_APPEND_DATA | GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
                    NULL, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (h == INVALID_HANDLE_VALUE) return 0;
    memset(&ov, 0, sizeof(ov));
    if (!LockFileEx(h, LOCKFILE_EXCLUSIVE_LOCK, 0, MAXDWORD, MAXDWORD, &ov)) goto end;
    locked = 1;
    while (done < n) {
        DWORD chunk = (DWORD)((n - done) > 0x7ffff000u ? 0x7ffff000u : (n - done));
        DWORD wrote = 0;
        if (!WriteFile(h, buf + done, chunk, &wrote, NULL) || wrote == 0) goto end;
        done += wrote;
    }
    ok = FlushFileBuffers(h) != 0;
end:
    if (locked) UnlockFileEx(h, 0, MAXDWORD, MAXDWORD, &ov);
    CloseHandle(h);
    return ok;
#else
    int fd = open(path, O_WRONLY | O_CREAT | O_APPEND, 0666);
    size_t done = 0;
    if (fd < 0) return 0;
    while (done < n) {
        ssize_t wrote = write(fd, buf + done, n - done);
        if (wrote < 0 && errno == EINTR) continue;
        if (wrote <= 0) { close(fd); return 0; }
        done += (size_t)wrote;
    }
    if (fsync(fd) != 0) { close(fd); return 0; }
    return close(fd) == 0;
#endif
}

int ama10_ledger_append_event(const char *path, const AMA10ShardEvent *e) {
    char timestamp[32];
    char *line, *p;
    size_t need;
    time_t now;
    struct tm utc;
    if (!path || !*path || !e || !member(e->dataset_id, dataset_ids, sizeof(dataset_ids)/sizeof(dataset_ids[0])) ||
        !member(e->state, states, sizeof(states)/sizeof(states[0])) || !e->shard_id || !*e->shard_id ||
        !e->source_revision || !e->source_url || !e->local_path || !valid_hash(e->sha256) ||
        (e->processed_path && (!*e->processed_path || !valid_hash(e->processed_sha256))) ||
        (!e->processed_path && e->processed_bytes) ||
        e->validated_bytes > e->acquired_bytes || e->consumed_tokens > e->valid_tokens) return 0;
    now = time(NULL);
#ifdef _WIN32
    if (gmtime_s(&utc, &now) != 0) return 0;
#else
    if (!gmtime_r(&now, &utc)) return 0;
#endif
    if (strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%SZ", &utc) == 0) return 0;
    need = 512 + strlen(timestamp) + escaped_size(e->dataset_id) + escaped_size(e->shard_id) +
        escaped_size(e->state) + escaped_size(e->source_revision) + escaped_size(e->source_url) +
        escaped_size(e->local_path) + escaped_size(e->sha256) +
        (e->processed_path ? escaped_size(e->processed_path) : 0) +
        (e->processed_sha256 ? escaped_size(e->processed_sha256) : 0);
    if (need < 256) return 0;
    line = (char *)malloc(need);
    if (!line) return 0;
    p = line;
#define LIT(s) do { size_t z = sizeof(s)-1; memcpy(p, (s), z); p += z; } while (0)
#define STR(name, value, comma) do { LIT("\"" name "\":\""); p = append_escaped(p, value); LIT("\"" comma); } while (0)
    LIT("{\"schema_version\":1,\"record_type\":\"shard_event\",\"timestamp_utc\":\"");
    p = append_escaped(p, timestamp); LIT("\",");
    STR("dataset_id", e->dataset_id, ",");
    STR("shard_id", e->shard_id, ",");
    STR("state", e->state, ",");
    STR("source_revision", e->source_revision, ",");
    STR("source_url", e->source_url, ",");
    STR("local_path", e->local_path, ",");
    STR("sha256", e->sha256, ",");
    p += (size_t)sprintf(p, "\"acquired_bytes\":%llu,\"validated_bytes\":%llu,\"processed_bytes\":%llu,\"documents\":%llu,\"valid_tokens\":%llu,\"consumed_tokens\":%llu,\"processed_path\":",
        (unsigned long long)e->acquired_bytes, (unsigned long long)e->validated_bytes,
        (unsigned long long)e->processed_bytes,
        (unsigned long long)e->documents, (unsigned long long)e->valid_tokens,
        (unsigned long long)e->consumed_tokens);
    if (e->processed_path) {
        *p++ = '"'; p = append_escaped(p, e->processed_path); LIT("\",\"processed_sha256\":\"");
        p = append_escaped(p, e->processed_sha256); LIT("\"}\n");
    } else {
        LIT("null,\"processed_sha256\":null}\n");
    }
#undef STR
#undef LIT
    if ((size_t)(p - line) >= need) { free(line); return 0; }
    {
        int ok = write_all(path, line, (size_t)(p - line));
        free(line);
        return ok;
    }
}
