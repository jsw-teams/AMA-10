#ifndef AMA10_TOKENIZER_H
#define AMA10_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint32_t left;
    uint32_t right;
    unsigned char *bytes;
    size_t length;
    uint32_t special;
} AMA10Token;

typedef struct {
    AMA10Token *tokens;
    size_t vocabulary_size;
    size_t capacity;
} AMA10Tokenizer;

/* Zero-initialize before first use. Byte fallback IDs 0..255 preserve any byte sequence. */
int ama10_tokenizer_train(AMA10Tokenizer *t, const unsigned char *text,
                          size_t length, size_t target_vocabulary);
/* Append merge tokens without changing existing token IDs or merge ranks. */
int ama10_tokenizer_extend(AMA10Tokenizer *t,const unsigned char *text,
                           size_t length,size_t target_vocabulary);
/* Append a named non-text token. Normal byte encoding never emits this ID. */
int ama10_tokenizer_add_special(AMA10Tokenizer *t,const unsigned char *bytes,
                                size_t length,uint32_t *token_id);
int ama10_tokenizer_encode(const AMA10Tokenizer *t, const unsigned char *text,
                           size_t length, uint32_t **ids, size_t *count);
int ama10_tokenizer_decode(const AMA10Tokenizer *t, const uint32_t *ids,
                           size_t count, unsigned char **text, size_t *length);
int ama10_tokenizer_save(const AMA10Tokenizer *t, const char *path);
int ama10_tokenizer_load(AMA10Tokenizer *t, const char *path);
void ama10_tokenizer_free(AMA10Tokenizer *t);

#endif
