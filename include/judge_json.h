#ifndef AMA10_JUDGE_JSON_H
#define AMA10_JUDGE_JSON_H
#include <stddef.h>

int ama10_json_string_field(const unsigned char *json,size_t json_bytes,
                            const char *field,char *text,size_t text_capacity);
int ama10_parse_preference_judgment(const char *json_text,char *choice,
                                    size_t choice_capacity,int *score_a,
                                    int *score_b);

#endif
