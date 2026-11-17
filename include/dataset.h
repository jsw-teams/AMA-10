#ifndef AMA10_DATASET_H
#define AMA10_DATASET_H

#include "tokenizer.h"
#include <stdint.h>
#include <stdio.h>

typedef struct {
    FILE *file;
    uint64_t token_count;
    uint64_t raw_bytes;
    uint64_t raw_hash;
    uint64_t tokenizer_hash;
    uint32_t vocabulary;
    uint32_t token_width;
} AMA10TokenDataset;

int ama10_file_fingerprint(const char *path,uint64_t *hash,uint64_t *byte_count);
uint64_t ama10_tokenizer_fingerprint(const AMA10Tokenizer *tokenizer);
int ama10_token_dataset_build(const char *raw_path,const char *cache_path,
                              const AMA10Tokenizer *tokenizer,
                              uint64_t raw_hash,uint64_t raw_bytes);
int ama10_token_dataset_open(AMA10TokenDataset *dataset,const char *cache_path,
                             const AMA10Tokenizer *tokenizer,
                             uint64_t expected_hash,uint64_t expected_bytes);
int ama10_token_dataset_build_or_open(AMA10TokenDataset *dataset,
                                      const char *raw_path,const char *cache_path,
                                      const AMA10Tokenizer *tokenizer,
                                      uint64_t raw_hash,uint64_t raw_bytes);
int ama10_token_dataset_read(AMA10TokenDataset *dataset,uint64_t cursor,
                             size_t count,uint32_t *tokens);
void ama10_token_dataset_close(AMA10TokenDataset *dataset);

#endif
