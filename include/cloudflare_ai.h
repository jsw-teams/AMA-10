#ifndef AMA10_CLOUDFLARE_AI_H
#define AMA10_CLOUDFLARE_AI_H
#include <stddef.h>

#define AMA10_CLOUDFLARE_TEACHER_MODEL "@cf/openai/gpt-oss-20b"

int ama10_cloudflare_response_text(const unsigned char *json,size_t json_bytes,
                                   char *text,size_t text_capacity);
char *ama10_cloudflare_read_token_file(const char *path);
int ama10_cloudflare_teacher_model_allowed(const char *model);
int ama10_cloudflare_generate(const char *prompt,const char *account_id,
                              const char *model,const char *api_token,
                              char *response,size_t response_capacity,
                              char *error,size_t error_capacity);

#endif
