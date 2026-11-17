#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winhttp.h>
#include <bcrypt.h>

#include "pipeline_ledger.h"
#include "shard_fetch.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

#define FETCH_BUFFER 65536u
#define FETCH_ATTEMPTS 5u

static DWORD s_min_delay = 30;
static ULONGLONG s_last_request_ms = 0;

static wchar_t *wide(const char *s) {
    int n;
    wchar_t *out;
    if (!s) return NULL;
    n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s, -1, NULL, 0);
    if (n <= 0) return NULL;
    out = (wchar_t *)malloc((size_t)n * sizeof(wchar_t));
    if (!out) return NULL;
    if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s, -1, out, n) != n) {
        free(out); return NULL;
    }
    return out;
}

static int ensure_parent(const char *path) {
    char *copy = _strdup(path), *p;
    if (!copy) return 0;
    for (p = copy + 1; *p; ++p) {
        if (*p == '\\' || *p == '/') {
            char hold = *p; *p = 0;
            if (*copy && !CreateDirectoryA(copy, NULL) && GetLastError() != ERROR_ALREADY_EXISTS) { free(copy); return 0; }
            *p = hold;
        }
    }
    free(copy); return 1;
}

static int file_size(const char *path, uint64_t *size) {
    HANDLE h = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                           NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    LARGE_INTEGER n;
    if (h == INVALID_HANDLE_VALUE) return 0;
    if (!GetFileSizeEx(h, &n) || n.QuadPart < 0) { CloseHandle(h); return 0; }
    CloseHandle(h); *size = (uint64_t)n.QuadPart; return 1;
}

static int read_meta(const char *path, char *validator, size_t cap) {
    FILE *f = fopen(path, "rb");
    size_t n;
    if (!f) return 0;
    n = fread(validator, 1, cap - 1, f); fclose(f);
    while (n && (validator[n-1] == '\r' || validator[n-1] == '\n')) --n;
    validator[n] = 0;
    return n > 0;
}

static int write_meta(const char *path, const wchar_t *validator) {
    int n = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, validator, -1, NULL, 0, NULL, NULL);
    char *utf8;
    FILE *f;
    if (n <= 1 || n > 2048) return 0;
    utf8 = (char *)malloc((size_t)n + 2);
    if (!utf8) return 0;
    if (WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, validator, -1, utf8, n, NULL, NULL) != n) { free(utf8); return 0; }
    f = fopen(path, "wb");
    if (!f) { free(utf8); return 0; }
    if (fwrite(utf8, 1, (size_t)n - 1, f) != (size_t)n - 1 || fputs("\n", f) < 0 || fflush(f) != 0) {
        fclose(f); free(utf8); return 0;
    }
    free(utf8); return fclose(f) == 0;
}

static int query_header(HINTERNET request, DWORD query, wchar_t *buffer, DWORD cap) {
    DWORD n = cap * sizeof(wchar_t), index = 0;
    if (!WinHttpQueryHeaders(request, query, WINHTTP_HEADER_NAME_BY_INDEX, buffer, &n, &index)) {
        if (GetLastError() == ERROR_INSUFFICIENT_BUFFER) return 0;
        buffer[0] = 0; return 0;
    }
    return 1;
}

static DWORD status_code(HINTERNET request) {
    DWORD status = 0, n = sizeof(status), index = 0;
    (void)WinHttpQueryHeaders(request, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                              WINHTTP_HEADER_NAME_BY_INDEX, &status, &n, &index);
    return status;
}

static DWORD retry_delay(HINTERNET request, unsigned attempt) {
    wchar_t value[256];
    DWORD wait = (DWORD)(30u << (attempt > 5 ? 5 : attempt));
    if (wait < s_min_delay) wait = s_min_delay;
    value[0] = 0;
    if (query_header(request, WINHTTP_QUERY_RETRY_AFTER, value, 256)) {
        wchar_t *end = NULL;
        unsigned long sec = wcstoul(value, &end, 10);
        if (end && *end == 0 && sec > wait) wait = (DWORD)(sec > 3600 ? 3600 : sec);
        else {
            SYSTEMTIME st; FILETIME ft; ULARGE_INTEGER when, now;
            if (WinHttpTimeToSystemTime(value, &st) && SystemTimeToFileTime(&st, &ft)) {
                when.LowPart = ft.dwLowDateTime; when.HighPart = ft.dwHighDateTime;
                GetSystemTimeAsFileTime(&ft); now.LowPart = ft.dwLowDateTime; now.HighPart = ft.dwHighDateTime;
                if (when.QuadPart > now.QuadPart) {
                    ULONGLONG delta = (when.QuadPart - now.QuadPart) / 10000000ULL;
                    if (delta > wait) wait = (DWORD)(delta > 3600 ? 3600 : delta);
                }
            }
        }
    }
    value[0] = 0;
    {
        DWORD n = sizeof(value), index = 0;
        if (WinHttpQueryHeaders(request, WINHTTP_QUERY_CUSTOM, L"RateLimit", value, &n, &index)) {
            wchar_t *t = wcsstr(value, L"t=");
            if (t) {
                wchar_t *end = NULL;
                unsigned long sec = wcstoul(t + 2, &end, 10);
                if (end != t + 2 && sec > wait) wait = (DWORD)(sec > 3600 ? 3600 : sec);
            }
        }
    }
    return wait;
}

static void wait_spacing(DWORD extra) {
    ULONGLONG now = GetTickCount64();
    if (s_last_request_ms) {
        ULONGLONG elapsed = now - s_last_request_ms;
        DWORD wait = extra > s_min_delay ? extra : s_min_delay;
        ULONGLONG target = (ULONGLONG)wait * 1000ULL;
        if (elapsed < target) Sleep((DWORD)(target - elapsed));
    } else if (extra) Sleep(extra * 1000u);
    s_last_request_ms = GetTickCount64();
}

static int sha256_file(const char *path, char out[65], uint64_t *bytes) {
    BCRYPT_ALG_HANDLE alg = NULL; BCRYPT_HASH_HANDLE hash = NULL;
    DWORD object_size = 0, got = 0, hash_size = 0;
    unsigned char *object = NULL, digest[32], buf[FETCH_BUFFER];
    FILE *f = fopen(path, "rb"); size_t n; int ok = 0;
    static const char hex[] = "0123456789abcdef";
    uint64_t total = 0; size_t i;
    if (!f) return 0;
    if (BCryptOpenAlgorithmProvider(&alg, BCRYPT_SHA256_ALGORITHM, NULL, 0) < 0 ||
        BCryptGetProperty(alg, BCRYPT_OBJECT_LENGTH, (PUCHAR)&object_size, sizeof(object_size), &got, 0) < 0 ||
        BCryptGetProperty(alg, BCRYPT_HASH_LENGTH, (PUCHAR)&hash_size, sizeof(hash_size), &got, 0) < 0 || hash_size != 32) goto done;
    object = (unsigned char *)malloc(object_size); if (!object) goto done;
    if (BCryptCreateHash(alg, &hash, object, object_size, NULL, 0, 0) < 0) goto done;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) {
        if (UINT64_MAX - total < n || BCryptHashData(hash, buf, (ULONG)n, 0) < 0) goto done;
        total += n;
    }
    if (ferror(f) || BCryptFinishHash(hash, digest, sizeof(digest), 0) < 0) goto done;
    for (i = 0; i < sizeof(digest); ++i) { out[i*2] = hex[digest[i] >> 4]; out[i*2+1] = hex[digest[i] & 15]; }
    out[64] = 0; *bytes = total; ok = 1;
done:
    if (hash) BCryptDestroyHash(hash);
    if (alg) BCryptCloseAlgorithmProvider(alg, 0);
    free(object); fclose(f); return ok;
}

static int valid_hf_revision(const char *url) {
    const char *p = strstr(url, "/resolve/"); size_t i;
    if (!p) return 1;
    p += 9;
    for (i = 0; i < 40; ++i) if (!p[i] || !((p[i] >= '0' && p[i] <= '9') || (p[i] >= 'a' && p[i] <= 'f'))) return 0;
    return p[40] == '/';
}

static int valid_expected_sha(const char *s) {
    size_t i;
    if (!s) return 0;
    if (strcmp(s, "-") == 0) return 1;
    if (strlen(s) != 64) return 0;
    for (i = 0; i < 64; ++i) if (!((s[i] >= '0' && s[i] <= '9') || (s[i] >= 'a' && s[i] <= 'f') || (s[i] >= 'A' && s[i] <= 'F'))) return 0;
    return 1;
}

static int record_download(const AMA10FetchRequest *r, const char *sha, uint64_t bytes) {
    AMA10ShardEvent e;
    e.dataset_id = r->dataset_id; e.shard_id = r->shard_id; e.state = "downloaded";
    e.source_revision = r->source_revision; e.source_url = r->url; e.local_path = r->output_path;
    e.sha256 = sha; e.processed_path = NULL; e.processed_sha256 = NULL;
    e.acquired_bytes = bytes; e.validated_bytes = 0; e.processed_bytes = 0;
    e.documents = 0; e.valid_tokens = 0; e.consumed_tokens = 0;
    return ama10_ledger_append_event(r->ledger_path, &e);
}

int ama10_fetch_official_shard(const AMA10FetchRequest *r, char sha[65], uint64_t *downloaded) {
    URL_COMPONENTS uc; wchar_t *wurl = NULL, host[512], path[8192];
    DWORD status = 0; unsigned attempt; char *part = NULL, *meta = NULL;
    uint64_t final_bytes = 0; char validator_utf8[2048];
    HINTERNET session = NULL, connection = NULL, req = NULL;
    int ok = 0; DWORD retry_wait = 0;
    if (!r || !sha || !downloaded || !r->ledger_path || !r->dataset_id || !r->shard_id || !r->source_revision ||
        !r->url || !r->output_path || !valid_expected_sha(r->expected_sha256) || !r->max_bytes || r->minimum_delay_seconds < 30 ||
        !valid_hf_revision(r->url) || !ensure_parent(r->output_path)) return 0;
    s_min_delay = r->minimum_delay_seconds;
    part = (char *)malloc(strlen(r->output_path) + 9); meta = (char *)malloc(strlen(r->output_path) + 14);
    if (!part || !meta) goto cleanup;
    sprintf(part, "%s.partial", r->output_path); sprintf(meta, "%s.partial.meta", r->output_path);
    if (file_size(r->output_path, &final_bytes)) {
        if (final_bytes > r->max_bytes || strcmp(r->expected_sha256, "-") == 0 ||
            !sha256_file(r->output_path, sha, &final_bytes) || _stricmp(sha, r->expected_sha256) != 0 ||
            !record_download(r, sha, final_bytes)) goto cleanup;
        *downloaded = final_bytes; ok = 1; goto cleanup;
    }
    wurl = wide(r->url); if (!wurl) goto cleanup;
    memset(&uc, 0, sizeof(uc)); uc.dwStructSize = sizeof(uc);
    uc.lpszHostName = host; uc.dwHostNameLength = sizeof(host)/sizeof(host[0]);
    uc.lpszUrlPath = path; uc.dwUrlPathLength = sizeof(path)/sizeof(path[0]);
    if (!WinHttpCrackUrl(wurl, 0, 0, &uc) || uc.nScheme != INTERNET_SCHEME_HTTPS) goto cleanup;
    {
        const char *commit = strstr(r->url, "/resolve/");
        if (commit && (strlen(r->source_revision) != 40 || strncmp(commit + 9, r->source_revision, 40) != 0 || commit[49] != '/')) goto cleanup;
    }
    host[uc.dwHostNameLength] = 0; path[uc.dwUrlPathLength] = 0;
    if (uc.dwExtraInfoLength && wcslen(path) + uc.dwExtraInfoLength < sizeof(path)/sizeof(path[0]))
        wcsncat(path, uc.lpszExtraInfo, uc.dwExtraInfoLength);
    session = WinHttpOpen(L"AMA-10 official shard fetch/1.0 (C11; polite serial client)",
                          WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY, WINHTTP_NO_PROXY_NAME,
                          WINHTTP_NO_PROXY_BYPASS, 0);
    if (!session) goto cleanup;
    (void)WinHttpSetTimeouts(session, 15000, 15000, 30000, 120000);
    connection = WinHttpConnect(session, host, uc.nPort, 0); if (!connection) goto cleanup;
    for (attempt = 0; attempt < FETCH_ATTEMPTS; ++attempt) {
        wchar_t headers[4096]; DWORD flags = WINHTTP_FLAG_SECURE;
        FILE *out = NULL; uint64_t partial_size = 0, written = 0;
        wchar_t *ifrange = NULL; DWORD status_now;
        if (file_size(part, &partial_size) && partial_size && read_meta(meta, validator_utf8, sizeof(validator_utf8))) {
            ifrange = wide(validator_utf8); if (!ifrange) goto cleanup;
        } else {
            partial_size = 0; DeleteFileA(part); DeleteFileA(meta);
        }
        req = WinHttpOpenRequest(connection, L"GET", path, NULL, WINHTTP_NO_REFERER,
                                 WINHTTP_DEFAULT_ACCEPT_TYPES, flags);
        if (!req) { free(ifrange); goto cleanup; }
        headers[0] = 0;
        if (partial_size) {
            _snwprintf(headers, sizeof(headers)/sizeof(headers[0])-1,
                       L"Range: bytes=%llu-\r\nIf-Range: %s\r\n", (unsigned long long)partial_size, ifrange);
            headers[sizeof(headers)/sizeof(headers[0])-1] = 0;
        }
        free(ifrange);
        wait_spacing(retry_wait); retry_wait = 0;
        if (!WinHttpSendRequest(req, headers[0] ? headers : WINHTTP_NO_ADDITIONAL_HEADERS,
                                headers[0] ? (DWORD)-1L : 0, WINHTTP_NO_REQUEST_DATA, 0, 0, 0) ||
            !WinHttpReceiveResponse(req, NULL)) {
            DWORD delay = (DWORD)(30u << (attempt > 5 ? 5 : attempt));
            WinHttpCloseHandle(req); req = NULL;
            if (attempt + 1 == FETCH_ATTEMPTS) goto cleanup;
            retry_wait = delay; continue;
        }
        status_now = status_code(req); status = status_now;
        if (status_now == 429 || status_now == 503 || status_now == 502 || status_now == 504 || status_now == 408) {
            DWORD delay = retry_delay(req, attempt);
            WinHttpCloseHandle(req); req = NULL;
            if (attempt + 1 == FETCH_ATTEMPTS) goto cleanup;
            retry_wait = delay;
            continue;
        }
        if (status_now == 416 && partial_size) {
            WinHttpCloseHandle(req); req = NULL; DeleteFileA(part); DeleteFileA(meta); continue;
        }
        if (status_now != 200 && status_now != 206) { WinHttpCloseHandle(req); req = NULL; goto cleanup; }
        if (status_now == 206 && !partial_size) { WinHttpCloseHandle(req); req = NULL; goto cleanup; }
        if (status_now == 200 && partial_size) { partial_size = 0; DeleteFileA(part); DeleteFileA(meta); }
        if (status_now == 206) {
            wchar_t range_header[256]; wchar_t *end = NULL; unsigned long long start;
            if (!query_header(req, WINHTTP_QUERY_CONTENT_RANGE, range_header, 256) ||
                wcsncmp(range_header, L"bytes ", 6) != 0) { WinHttpCloseHandle(req); req = NULL; goto cleanup; }
            start = wcstoull(range_header + 6, &end, 10);
            if (!end || end == range_header + 6 || *end != L'-' || start != (unsigned long long)partial_size) {
                WinHttpCloseHandle(req); req = NULL; goto cleanup;
            }
        }
        {
            wchar_t etag[2048];
            if (!query_header(req, WINHTTP_QUERY_ETAG, etag, 2048))
                (void)query_header(req, WINHTTP_QUERY_LAST_MODIFIED, etag, 2048);
            if (etag[0]) (void)write_meta(meta, etag);
        }
        if (status_now == 206) out = fopen(part, "ab"); else out = fopen(part, "wb");
        if (!out) { WinHttpCloseHandle(req); req = NULL; goto cleanup; }
        for (;;) {
            DWORD avail = 0, got = 0;
            if (!WinHttpQueryDataAvailable(req, &avail)) break;
            if (!avail) { ok = 1; break; }
            if (avail > FETCH_BUFFER) avail = FETCH_BUFFER;
            {
                unsigned char buffer[FETCH_BUFFER];
                if (!WinHttpReadData(req, buffer, avail, &got) || got == 0) break;
                if (partial_size > r->max_bytes || written > r->max_bytes - partial_size ||
                    got > r->max_bytes - partial_size - written ||
                    fwrite(buffer, 1, got, out) != got) { ok = 0; break; }
                written += got;
            }
        }
        if (fflush(out) != 0) ok = 0;
        if (fclose(out) != 0) ok = 0;
        WinHttpCloseHandle(req); req = NULL;
        if (!ok) { ok = 0; goto cleanup; }
        if (!sha256_file(part, sha, &final_bytes) || final_bytes == 0 || final_bytes > r->max_bytes) { ok = 0; goto cleanup; }
        if (strcmp(r->expected_sha256, "-") != 0 && _stricmp(sha, r->expected_sha256) != 0) {
            DeleteFileA(part); DeleteFileA(meta); ok = 0;
            if (attempt + 1 < FETCH_ATTEMPTS) { retry_wait = s_min_delay; continue; }
            goto cleanup;
        }
        if (!MoveFileExA(part, r->output_path, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) { ok = 0; goto cleanup; }
        DeleteFileA(meta);
        if (!record_download(r, sha, final_bytes)) { ok = 0; goto cleanup; }
        *downloaded = final_bytes; ok = 1; goto cleanup;
    }
cleanup:
    if (req) WinHttpCloseHandle(req);
    if (connection) WinHttpCloseHandle(connection);
    if (session) WinHttpCloseHandle(session);
    free(wurl); free(part); free(meta);
    (void)status;
    return ok;
}
