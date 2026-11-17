#ifndef AMA10_EVAL_H
#define AMA10_EVAL_H

#include "tokenizer.h"

typedef struct {
    double distinct_token_ratio;
    double repeated_fourgram_ratio;
    uint64_t codepoints;
    uint64_t invalid_utf8_bytes;
    uint64_t latin,cjk,kana,hangul,cyrillic,arabic,devanagari,bengali,other;
} AMA10GenerationMetrics;

int ama10_generation_metrics(const AMA10Tokenizer *tokenizer,
                             const uint32_t *tokens,size_t token_count,
                             const unsigned char *bytes,size_t byte_count,
                             AMA10GenerationMetrics *metrics);

int ama10_text_error_rates(const unsigned char *reference,size_t reference_bytes,
                           const unsigned char *hypothesis,size_t hypothesis_bytes,
                           double *wer,double *cer,uint64_t *word_errors,
                           uint64_t *reference_words,uint64_t *character_errors,
                           uint64_t *reference_characters);

#endif
