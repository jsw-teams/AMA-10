#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <bcrypt.h>

#include "pipeline_ledger.h"
#include "duckdb.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define STORAGE_LIMIT (UINT64_C(100) * 1024 * 1024 * 1024)
#define HASH_OFFSET UINT64_C(14695981039346656037)
#define HASH_PRIME UINT64_C(1099511628211)
#define IO_CHUNK 65536u

typedef struct { uint64_t hash, offset, length; unsigned used; } DedupEntry;
typedef struct { DedupEntry *items; size_t capacity, count; } DedupSet;
typedef struct { char code[64]; uint64_t rows, accepted; } LanguageCount;
typedef struct {
    duckdb_state (*open)(const char *, duckdb_database *);
    duckdb_state (*connect)(duckdb_database, duckdb_connection *);
    void (*disconnect)(duckdb_connection *);
    void (*close)(duckdb_database *);
    duckdb_state (*query)(duckdb_connection, const char *, duckdb_result *);
    duckdb_state (*prepare)(duckdb_connection, const char *, duckdb_prepared_statement *);
    const char *(*prepare_error)(duckdb_prepared_statement);
    duckdb_state (*bind_varchar)(duckdb_prepared_statement, idx_t, const char *);
    duckdb_state (*bind_double)(duckdb_prepared_statement, idx_t, double);
    duckdb_state (*execute_prepared_streaming)(duckdb_prepared_statement, duckdb_result *);
    void (*destroy_prepare)(duckdb_prepared_statement *);
    bool (*result_is_streaming)(duckdb_result);
    duckdb_data_chunk (*fetch_chunk)(duckdb_result);
    void (*destroy_result)(duckdb_result *);
    const char *(*result_error)(duckdb_result *);
    void (*destroy_chunk)(duckdb_data_chunk *);
    idx_t (*chunk_size)(duckdb_data_chunk);
    duckdb_vector (*chunk_vector)(duckdb_data_chunk, idx_t);
    void *(*vector_data)(duckdb_vector);
    uint64_t *(*vector_validity)(duckdb_vector);
    bool (*validity_row)(uint64_t *, idx_t);
    uint32_t (*string_length)(duckdb_string_t);
    const char *(*string_data)(duckdb_string_t *);
} DuckApi;

typedef struct {
    uint64_t rows, empty, invalid_utf8, quality_rejected, duplicates, accepted;
    double score_min, score_max, score_sum; uint64_t score_count;
    LanguageCount languages[128]; size_t language_count;
} ProcessStats;

static DuckApi api;
static HMODULE duckdb_module;

#define LOAD(field, symbol) do { FARPROC proc = GetProcAddress(duckdb_module, symbol); if (!proc || sizeof(proc) != sizeof(api.field)) return 0; memcpy(&api.field, &proc, sizeof(proc)); } while (0)
typedef duckdb_state (*F_open)(const char *, duckdb_database *);
typedef duckdb_state (*F_connect)(duckdb_database, duckdb_connection *);
typedef void (*F_disconnect)(duckdb_connection *);
typedef void (*F_close)(duckdb_database *);
typedef duckdb_state (*F_query)(duckdb_connection, const char *, duckdb_result *);
typedef duckdb_state (*F_prepare)(duckdb_connection, const char *, duckdb_prepared_statement *);
typedef const char *(*F_prepare_error)(duckdb_prepared_statement);
typedef duckdb_state (*F_bind_varchar)(duckdb_prepared_statement, idx_t, const char *);
typedef duckdb_state (*F_bind_double)(duckdb_prepared_statement, idx_t, double);
typedef duckdb_state (*F_execute_prepared_streaming)(duckdb_prepared_statement, duckdb_result *);
typedef void (*F_destroy_prepare)(duckdb_prepared_statement *);
typedef bool (*F_result_is_streaming)(duckdb_result);
typedef duckdb_data_chunk (*F_fetch_chunk)(duckdb_result);
typedef void (*F_destroy_result)(duckdb_result *);
typedef const char *(*F_result_error)(duckdb_result *);
typedef void (*F_destroy_chunk)(duckdb_data_chunk *);
typedef idx_t (*F_chunk_size)(duckdb_data_chunk);
typedef duckdb_vector (*F_chunk_vector)(duckdb_data_chunk, idx_t);
typedef void *(*F_vector_data)(duckdb_vector);
typedef uint64_t *(*F_vector_validity)(duckdb_vector);
typedef bool (*F_validity_row)(uint64_t *, idx_t);
typedef uint32_t (*F_string_length)(duckdb_string_t);
typedef const char *(*F_string_data)(duckdb_string_t *);

static int load_duckdb(void) {
    duckdb_module = LoadLibraryA("duckdb.dll");
    if (!duckdb_module) return 0;
    LOAD(open,"duckdb_open"); LOAD(connect,"duckdb_connect"); LOAD(disconnect,"duckdb_disconnect");
    LOAD(close,"duckdb_close"); LOAD(query,"duckdb_query"); LOAD(prepare,"duckdb_prepare");
    LOAD(prepare_error,"duckdb_prepare_error"); LOAD(bind_varchar,"duckdb_bind_varchar");
    LOAD(bind_double,"duckdb_bind_double");
    LOAD(execute_prepared_streaming,"duckdb_execute_prepared_streaming");
    LOAD(destroy_prepare,"duckdb_destroy_prepare"); LOAD(result_is_streaming,"duckdb_result_is_streaming");
    LOAD(fetch_chunk,"duckdb_fetch_chunk"); LOAD(destroy_result,"duckdb_destroy_result");
    LOAD(result_error,"duckdb_result_error"); LOAD(destroy_chunk,"duckdb_destroy_data_chunk");
    LOAD(chunk_size,"duckdb_data_chunk_get_size"); LOAD(chunk_vector,"duckdb_data_chunk_get_vector");
    LOAD(vector_data,"duckdb_vector_get_data"); LOAD(vector_validity,"duckdb_vector_get_validity");
    LOAD(validity_row,"duckdb_validity_row_is_valid"); LOAD(string_length,"duckdb_string_t_length");
    LOAD(string_data,"duckdb_string_t_data");
    return 1;
}

static int valid_utf8(const unsigned char *s, size_t n) {
    size_t i = 0;
    while (i < n) {
        unsigned c = s[i++];
        if (c < 0x80) { if (c == 0) return 0; continue; }
        if (c >= 0xc2 && c <= 0xdf) {
            if (i >= n || (s[i++] & 0xc0) != 0x80) return 0;
        } else if (c >= 0xe0 && c <= 0xef) {
            unsigned a, b; if (i + 1 >= n) return 0; a=s[i++]; b=s[i++];
            if ((b & 0xc0) != 0x80 || (a & 0xc0) != 0x80 || (c==0xe0 && a<0xa0) || (c==0xed && a>=0xa0)) return 0;
        } else if (c >= 0xf0 && c <= 0xf4) {
            unsigned a,b,d; if (i + 2 >= n) return 0; a=s[i++]; b=s[i++]; d=s[i++];
            if ((a&0xc0)!=0x80 || (b&0xc0)!=0x80 || (d&0xc0)!=0x80 ||
                (c==0xf0 && a<0x90) || (c==0xf4 && a>=0x90)) return 0;
        } else return 0;
    }
    return 1;
}

static int normalize(const char *src, size_t n, char **out, size_t *out_n) {
    char *p; size_t i, k=0; int space=1;
    if (!valid_utf8((const unsigned char *)src,n) || n==0 || n>SIZE_MAX-1) return 0;
    p=(char *)malloc(n+1); if(!p) return -1;
    for(i=0;i<n;i++) {
        unsigned char c=(unsigned char)src[i];
        if(c=='\r'||c=='\n'||c=='\t'||c==0x0b||c==0x0c) c=' ';
        if(c<0x20 && c!=' ') c=' ';
        if(c==' ') { if(!space) {p[k++]=' ';space=1;} }
        else {p[k++]=(char)c;space=0;}
    }
    if(k && p[k-1]==' ') --k;
    if(!k){free(p);return 0;}
    p[k]=0;*out=p;*out_n=k;return 1;
}

static uint64_t hash_text(const char *p,size_t n) {
    uint64_t h=HASH_OFFSET;size_t i;
    for(i=0;i<n;i++){h^=(unsigned char)p[i];h*=HASH_PRIME;}
    return h;
}

static int seek64(FILE *f,uint64_t pos) {
    if(pos>(uint64_t)INT64_MAX)return 0;
    return _fseeki64(f,(__int64)pos,SEEK_SET)==0;
}

static int same_at(FILE *f,uint64_t offset,const char *text,size_t n) {
    unsigned char buf[IO_CHUNK];size_t done=0;int equal=1;
    if(!seek64(f,offset))return 0;
    while(done<n){size_t take=n-done;if(take>sizeof(buf))take=sizeof(buf);
        if(fread(buf,1,take,f)!=take||memcmp(buf,text+done,take)!=0){equal=0;break;}done+=take;}
    (void)_fseeki64(f,0,SEEK_END);return equal;
}

static int dedup_grow(DedupSet *s) {
    size_t cap=s->capacity?s->capacity*2:1024,i;DedupEntry *p;
    if(cap<s->capacity||cap>SIZE_MAX/sizeof(*p))return 0;
    p=(DedupEntry *)calloc(cap,sizeof(*p));if(!p)return 0;
    for(i=0;i<s->capacity;i++)if(s->items[i].used){size_t j=(size_t)s->items[i].hash&(cap-1);
        while(p[j].used)j=(j+1)&(cap-1);
        p[j]=s->items[i];}
    free(s->items);s->items=p;s->capacity=cap;return 1;
}

static int dedup_check_add(DedupSet *s,FILE *out,const char *text,size_t n,uint64_t *hash_out) {
    uint64_t h=hash_text(text,n);size_t i;
    if(!s->capacity||((s->count+1)*10>s->capacity*7))if(!dedup_grow(s))return -1;
    i=(size_t)h&(s->capacity-1);
    while(s->items[i].used){
        DedupEntry *e=&s->items[i];
        if(e->hash==h&&e->length==n&&same_at(out,e->offset,text,n)){*hash_out=h;return 0;}
        i=(i+1)&(s->capacity-1);
    }
    {
        __int64 pos=_ftelli64(out);if(pos<0)return -1;
        s->items[i].used=1;s->items[i].hash=h;s->items[i].offset=(uint64_t)pos;s->items[i].length=n;s->count++;
    }
    *hash_out=h;return 1;
}

static int sha256_file(const char *path,char out[65],uint64_t *bytes) {
    BCRYPT_ALG_HANDLE alg=NULL;BCRYPT_HASH_HANDLE hash=NULL;DWORD object_size=0,got=0,hash_size=0;
    unsigned char *object=NULL,digest[32],buf[IO_CHUNK];FILE *f=fopen(path,"rb");size_t n,i;uint64_t total=0;int ok=0;
    static const char hex[]="0123456789abcdef";if(!f)return 0;
    if(BCryptOpenAlgorithmProvider(&alg,BCRYPT_SHA256_ALGORITHM,NULL,0)<0||
       BCryptGetProperty(alg,BCRYPT_OBJECT_LENGTH,(PUCHAR)&object_size,sizeof(object_size),&got,0)<0||
       BCryptGetProperty(alg,BCRYPT_HASH_LENGTH,(PUCHAR)&hash_size,sizeof(hash_size),&got,0)<0||hash_size!=32)goto done;
    object=(unsigned char *)malloc(object_size);if(!object)goto done;
    if(BCryptCreateHash(alg,&hash,object,object_size,NULL,0,0)<0)goto done;
    while((n=fread(buf,1,sizeof(buf),f))>0){if(UINT64_MAX-total<n||BCryptHashData(hash,buf,(ULONG)n,0)<0)goto done;total+=n;}
    if(ferror(f)||BCryptFinishHash(hash,digest,sizeof(digest),0)<0)goto done;
    for(i=0;i<sizeof(digest);i++){out[2*i]=hex[digest[i]>>4];out[2*i+1]=hex[digest[i]&15];}out[64]=0;*bytes=total;ok=1;
done:if(hash)BCryptDestroyHash(hash);if(alg)BCryptCloseAlgorithmProvider(alg,0);free(object);fclose(f);return ok;
}

static uint64_t tree_bytes(const char *path) {
    WIN32_FIND_DATAA fd;char pattern[MAX_PATH*4];HANDLE h;uint64_t sum=0;
    if(snprintf(pattern,sizeof(pattern),"%s\\*",path)<0||strlen(pattern)>=sizeof(pattern)-1)return UINT64_MAX;
    h=FindFirstFileA(pattern,&fd);if(h==INVALID_HANDLE_VALUE)return 0;
    do {
        char child[MAX_PATH*4];if(strcmp(fd.cFileName,".")==0||strcmp(fd.cFileName,"..")==0)continue;
        if(snprintf(child,sizeof(child),"%s\\%s",path,fd.cFileName)<0||strlen(child)>=sizeof(child)-1){sum=UINT64_MAX;break;}
        if(fd.dwFileAttributes&FILE_ATTRIBUTE_DIRECTORY){uint64_t x=tree_bytes(child);if(UINT64_MAX-sum<x){sum=UINT64_MAX;break;}sum+=x;}
        else {uint64_t x=((uint64_t)fd.nFileSizeHigh<<32)|fd.nFileSizeLow;if(UINT64_MAX-sum<x){sum=UINT64_MAX;break;}sum+=x;}
    } while(FindNextFileA(h,&fd));
    FindClose(h);return sum;
}

static int parquet_signature(const char *path,uint64_t *bytes) {
    FILE *f=fopen(path,"rb");unsigned char head[4],tail[4];int ok=0;__int64 n;
    if(!f)return 0;
    if(fread(head,1,4,f)!=4||memcmp(head,"PAR1",4)!=0||_fseeki64(f,0,SEEK_END)!=0||(n=_ftelli64(f))<8||
       _fseeki64(f,n-4,SEEK_SET)!=0||fread(tail,1,4,f)!=4||memcmp(tail,"PAR1",4)!=0)goto done;
    *bytes=(uint64_t)n;ok=1;
done:fclose(f);return ok;
}

static int ensure_parent(const char *path) {
    char *copy=_strdup(path),*p;if(!copy)return 0;
    for(p=copy+1;*p;p++)if(*p=='\\'||*p=='/'){
        char hold=*p;*p=0;if(*copy&&!CreateDirectoryA(copy,NULL)&&GetLastError()!=ERROR_ALREADY_EXISTS){free(copy);return 0;}*p=hold;
    }
    free(copy);return 1;
}

static size_t language_slot(ProcessStats *s,const char *p,size_t n) {
    size_t i;if(!n||n>=64){p="unknown";n=7;}
    for(i=0;i<s->language_count;i++)if(strlen(s->languages[i].code)==n&&memcmp(s->languages[i].code,p,n)==0)break;
    if(i==s->language_count){if(i>=sizeof(s->languages)/sizeof(s->languages[0]))return SIZE_MAX;
        memcpy(s->languages[i].code,p,n);s->languages[i].code[n]=0;s->language_count++;}
    return i;
}

static int write_stats(const char *path,const ProcessStats *s,uint64_t source_bytes,uint64_t output_bytes,
                       double min_score,const char *raw_sha,const char *output_sha) {
    char stats_path[MAX_PATH*4];FILE *f;size_t i;
    if(snprintf(stats_path,sizeof(stats_path),"%s.stats.json",path)<0||strlen(stats_path)>=sizeof(stats_path)-1)return 0;
    f=fopen(stats_path,"wb");if(!f)return 0;
    fprintf(f,"{\"source_bytes\":%llu,\"processed_bytes\":%llu,\"rows\":%llu,\"accepted_unique_documents\":%llu,\"duplicates_within_shard\":%llu,\"empty_documents\":%llu,\"invalid_utf8_documents\":%llu,\"quality_rejected\":%llu,\"quality_min_score\":%.6f,\"score_count\":%llu,\"score_min\":%.6f,\"score_max\":%.6f,\"score_mean\":%.6f,\"source_sha256\":\"%s\",\"processed_sha256\":\"%s\",\"languages\":[",
        (unsigned long long)source_bytes,(unsigned long long)output_bytes,(unsigned long long)s->rows,
        (unsigned long long)s->accepted,(unsigned long long)s->duplicates,(unsigned long long)s->empty,
        (unsigned long long)s->invalid_utf8,(unsigned long long)s->quality_rejected,min_score,
        (unsigned long long)s->score_count,s->score_count?s->score_min:0.0,s->score_count?s->score_max:0.0,
        s->score_count?s->score_sum/(double)s->score_count:0.0,raw_sha,output_sha);
    for(i=0;i<s->language_count;i++)fprintf(f,"%s{\"language\":\"%s\",\"rows\":%llu,\"accepted\":%llu}",i?",":"",s->languages[i].code,
        (unsigned long long)s->languages[i].rows,(unsigned long long)s->languages[i].accepted);
    fputs("]}\n",f);
    if(fflush(f)!=0){fclose(f);return 0;}return fclose(f)==0;
}

static void ledger_event(const char *ledger,const char *dataset,const char *shard,const char *revision,
                         const char *url,const char *raw_path,const char *raw_sha,const char *state,
                         uint64_t raw_bytes,uint64_t docs,const char *processed_path,
                         const char *processed_sha,uint64_t processed_bytes) {
    AMA10ShardEvent e;
    e.dataset_id=dataset;e.shard_id=shard;e.state=state;e.source_revision=revision;
    e.source_url=url;e.local_path=raw_path;e.sha256=raw_sha;e.processed_path=processed_path;
    e.processed_sha256=processed_sha;e.acquired_bytes=raw_bytes;
    e.validated_bytes=strcmp(state,"validated")==0||strcmp(state,"processed")==0?raw_bytes:0;
    e.processed_bytes=processed_bytes;e.documents=docs;e.valid_tokens=0;e.consumed_tokens=0;
    (void)ama10_ledger_append_event(ledger,&e);
}

int ama10_parquet_ingest(const char *raw,const char *output,const char *ledger,const char *dataset,
                         const char *shard,const char *revision,const char *url,const char *expected_sha,
                         double min_score) {
    char input_sha[65],output_sha[65],partial[MAX_PATH*4];uint64_t raw_bytes=0,hash_bytes_out=0,tree_start;
    uint64_t rows=0,out_bytes=0;FILE *out=NULL;duckdb_database db=NULL;duckdb_connection con=NULL;
    duckdb_prepared_statement stmt=NULL;duckdb_result result={0};
    DedupSet dedup={0};ProcessStats stats={0};int success=0,have_result=0;
    if(!raw||!output||!ledger||!dataset||!shard||!revision||!url||!expected_sha||
       strlen(output)+9>=sizeof(partial)||min_score < -1.0 || !isfinite(min_score)||!ensure_parent(output))return 0;
    if(!load_duckdb()){fprintf(stderr,"DuckDB C API DLL is unavailable (run scripts/fetch_duckdb.ps1).\n");goto done;}
    if(!parquet_signature(raw,&raw_bytes)||!sha256_file(raw,input_sha,&hash_bytes_out)||hash_bytes_out!=raw_bytes||
       (strcmp(expected_sha,"-")!=0&&_stricmp(input_sha,expected_sha)!=0)){
        fprintf(stderr,"Source Parquet signature, size, or expected SHA-256 validation failed.\n");goto done;
    }
    tree_start=tree_bytes("data");if(tree_start==UINT64_MAX||tree_start>STORAGE_LIMIT){fprintf(stderr,"Data directory already exceeds the 100 GiB working limit.\n");goto done;}
    sprintf(partial,"%s.partial",output);
    if(GetFileAttributesA(output)!=INVALID_FILE_ATTRIBUTES){fprintf(stderr,"Processed shard already exists; refusing to overwrite: %s\n",output);goto done;}
    (void)DeleteFileA(partial);
    ledger_event(ledger,dataset,shard,revision,url,raw,input_sha,"validated",raw_bytes,0,NULL,NULL,0);
    if(api.open(NULL,&db)!=DuckDBSuccess||api.connect(db,&con)!=DuckDBSuccess){fprintf(stderr,"DuckDB open/connect failed.\n");goto done;}
    if(api.query(con,"SET memory_limit='2GB'",NULL)!=DuckDBSuccess||api.query(con,"SET threads=2",NULL)!=DuckDBSuccess){fprintf(stderr,"DuckDB resource configuration failed.\n");goto done;}
    out=fopen(partial,"w+b");if(!out){fprintf(stderr,"Cannot create processed shard: %s\n",partial);goto done;}
    if(api.prepare(con,"SELECT CAST(text AS VARCHAR), CAST(language AS VARCHAR), CAST(score AS DOUBLE) FROM read_parquet(?) WHERE text IS NOT NULL",&stmt)!=DuckDBSuccess){
        fprintf(stderr,"DuckDB Parquet query preparation failed: %s\n",api.prepare_error(stmt));goto done;
    }
    if(api.bind_varchar(stmt,1,raw)!=DuckDBSuccess||api.execute_prepared_streaming(stmt,&result)!=DuckDBSuccess){
        fprintf(stderr,"DuckDB Parquet query failed: %s\n",api.result_error(&result));goto done;
    }
    have_result=1;
    if(!api.result_is_streaming(result)){fprintf(stderr,"DuckDB did not provide a streaming result; refusing a materialized scan.\n");goto done;}
    for(;;){
        duckdb_data_chunk chunk=api.fetch_chunk(result);idx_t chunk_rows,row;
        if(!chunk)break;
        chunk_rows=api.chunk_size(chunk);
        {
            duckdb_vector tv=api.chunk_vector(chunk,0),lv=api.chunk_vector(chunk,1),sv=api.chunk_vector(chunk,2);
            duckdb_string_t *texts=(duckdb_string_t *)api.vector_data(tv),*langs=(duckdb_string_t *)api.vector_data(lv);
            double *scores=(double *)api.vector_data(sv);
            uint64_t *text_valid=api.vector_validity(tv),*lang_valid=api.vector_validity(lv),*score_valid=api.vector_validity(sv);
            for(row=0;row<chunk_rows;row++){
                const char *src,*lang="unknown";size_t n,lang_n=7,slot;uint32_t len;char *clean=NULL;size_t clean_n=0;
                int norm,unique;uint64_t doc_hash;double score=0.0;__int64 pos;static const char sep[]="\n\n";
                rows++;stats.rows++;
                if(!text_valid||api.validity_row(text_valid,row)){
                    len=api.string_length(texts[row]);src=api.string_data(&texts[row]);n=(size_t)len;
                }else {stats.empty++;continue;}
                if(!src){stats.invalid_utf8++;continue;}
                if(!lang_valid||api.validity_row(lang_valid,row)){lang=api.string_data(&langs[row]);lang_n=api.string_length(langs[row]);if(!lang) {lang="unknown";lang_n=7;}}
                slot=language_slot(&stats,lang,lang_n);if(slot!=SIZE_MAX)stats.languages[slot].rows++;
                if(score_valid&&api.validity_row(score_valid,row)){score=scores[row];if(isfinite(score)){if(!stats.score_count||score<stats.score_min)stats.score_min=score;if(!stats.score_count||score>stats.score_max)stats.score_max=score;stats.score_sum+=score;stats.score_count++;}}
                if(min_score>=0.0&&score<min_score){stats.quality_rejected++;continue;}
                if(!valid_utf8((const unsigned char *)src,n)){stats.invalid_utf8++;continue;}
                norm=normalize(src,n,&clean,&clean_n);if(norm<0){fprintf(stderr,"Out of memory normalizing a source record.\n");api.destroy_chunk(&chunk);goto done;}
                if(norm==0){stats.empty++;continue;}
                unique=dedup_check_add(&dedup,out,clean,clean_n,&doc_hash);
                if(unique<0){free(clean);fprintf(stderr,"Deduplication index allocation or file seek failed.\n");api.destroy_chunk(&chunk);goto done;}
                if(unique==0){stats.duplicates++;free(clean);continue;}
                pos=_ftelli64(out);if(pos<0||tree_start>STORAGE_LIMIT||out_bytes>STORAGE_LIMIT-tree_start||clean_n+sizeof(sep)-1>STORAGE_LIMIT-tree_start-out_bytes){free(clean);fprintf(stderr,"Processing would exceed the 100 GiB data storage budget.\n");api.destroy_chunk(&chunk);goto done;}
                if(fwrite(clean,1,clean_n,out)!=clean_n||fwrite(sep,1,sizeof(sep)-1,out)!=sizeof(sep)-1){free(clean);fprintf(stderr,"Processed shard write failed.\n");api.destroy_chunk(&chunk);goto done;}
                free(clean);out_bytes+=(uint64_t)clean_n+sizeof(sep)-1;stats.accepted++;if(slot!=SIZE_MAX)stats.languages[slot].accepted++;
                (void)doc_hash;(void)pos;
            }
        }
        api.destroy_chunk(&chunk);
        if(fflush(out)!=0){fprintf(stderr,"Processed shard flush failed.\n");goto done;}
        if(rows&&rows%250000==0)fprintf(stderr,"parquet_rows=%llu unique_documents=%llu output_bytes=%llu\n",
            (unsigned long long)rows,(unsigned long long)stats.accepted,(unsigned long long)out_bytes);
    }
    {
        const char *err=api.result_error(&result);if(err&&*err){fprintf(stderr,"DuckDB streaming scan error: %s\n",err);goto done;}
    }
    if(fflush(out)!=0||fclose(out)!=0){out=NULL;fprintf(stderr,"Processed shard final close failed.\n");goto done;}out=NULL;
    if(!sha256_file(partial,output_sha,&hash_bytes_out)||hash_bytes_out!=out_bytes){fprintf(stderr,"Processed shard hash/size validation failed.\n");goto done;}
    if(!MoveFileExA(partial,output,MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH)){fprintf(stderr,"Could not atomically commit processed shard.\n");goto done;}
    if(!write_stats(output,&stats,raw_bytes,out_bytes,min_score,input_sha,output_sha)){fprintf(stderr,"Could not write per-language and quality processing statistics.\n");goto done;}
    {
        AMA10ShardEvent e;e.dataset_id=dataset;e.shard_id=shard;e.state="processed";e.source_revision=revision;
        e.source_url=url;e.local_path=raw;e.sha256=input_sha;e.processed_path=output;e.processed_sha256=output_sha;
        e.acquired_bytes=raw_bytes;e.validated_bytes=raw_bytes;e.processed_bytes=out_bytes;e.documents=stats.accepted;
        e.valid_tokens=0;e.consumed_tokens=0;
        if(!ama10_ledger_append_event(ledger,&e)){fprintf(stderr,"Could not commit processed event to status ledger.\n");goto done;}
    }
    printf("source_rows=%llu unique_documents=%llu duplicate_documents=%llu invalid_utf8=%llu empty=%llu quality_rejected=%llu processed_bytes=%llu valid_tokens=0 sha256=%s output=%s\n",
        (unsigned long long)stats.rows,(unsigned long long)stats.accepted,(unsigned long long)stats.duplicates,
        (unsigned long long)stats.invalid_utf8,(unsigned long long)stats.empty,(unsigned long long)stats.quality_rejected,
        (unsigned long long)out_bytes,output_sha,output);
    success=1;
done:
    if(out)fclose(out);
    if(have_result)api.destroy_result(&result);
    if(stmt)api.destroy_prepare(&stmt);
    if(con)api.disconnect(&con);
    if(db)api.close(&db);
    if(duckdb_module)FreeLibrary(duckdb_module);
    free(dedup.items);
    return success;
}
