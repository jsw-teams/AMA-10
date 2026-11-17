#include "parquet_ingest.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc,char **argv) {
    char *end=NULL;double min_score=-1.0;
    if(argc!=9&&argc!=10){
        fprintf(stderr,"Usage: %s <raw.parquet> <processed.txt> <ledger.jsonl> <dataset_id> <shard_id> <commit> <source_url> <expected_sha256|-> [minimum_score=-1]\n",argv[0]);
        return 2;
    }
    if(argc==10){errno=0;min_score=strtod(argv[9],&end);if(errno||!end||*end||!isfinite(min_score)||min_score < -1.0){fputs("minimum_score must be -1 (no filter) or a finite value >= 0.\n",stderr);return 2;}}
    return ama10_parquet_ingest(argv[1],argv[2],argv[3],argv[4],argv[5],argv[6],argv[7],argv[8],min_score)?0:1;
}
