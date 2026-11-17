#include "judge_json.h"
#include "cloudflare_ai.h"
#include <stdio.h>
#include <string.h>
int main(void){char text[256],choice[8];int a=0,b=0;
    if(!ama10_cloudflare_teacher_model_allowed(AMA10_CLOUDFLARE_TEACHER_MODEL)||ama10_cloudflare_teacher_model_allowed("@cf/moonshotai/kimi-k2.7-code")||ama10_cloudflare_teacher_model_allowed("@cf/deepseek-ai/deepseek-r1-distill-qwen-32b")||ama10_cloudflare_teacher_model_allowed("@cf/qwen/qwen2.5-coder-32b-instruct"))goto fail;
    static const unsigned char cloudflare[]={"{\"success\":true,\"result\":{\"response\":\"{\\\"choice\\\":\\\"B\\\",\\\"score_a\\\":2,\\\"score_b\\\":5,\\\"reason\\\":\\\"better\\\"}\",\"usage\":{\"completion_tokens\":42}}}"};
    if(!ama10_cloudflare_response_text(cloudflare,sizeof(cloudflare)-1,text,sizeof(text))||!ama10_parse_preference_judgment(text,choice,sizeof(choice),&a,&b)||strcmp(choice,"B")||a!=2||b!=5)goto fail;
    static const unsigned char cloudflare_chat[]={"{\"result\":{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"{\\\"choice\\\":\\\"A\\\",\\\"score_a\\\":5,\\\"score_b\\\":2,\\\"reason\\\":\\\"reference\\\"}\"}}]}}"};
    if(!ama10_cloudflare_response_text(cloudflare_chat,sizeof(cloudflare_chat)-1,text,sizeof(text))||!ama10_parse_preference_judgment(text,choice,sizeof(choice),&a,&b)||strcmp(choice,"A")||a!=5||b!=2)goto fail;
    puts("Cloudflare Workers AI response decoding, preference schema, and teacher model allowlist: ok");return 0;
fail:fprintf(stderr,"Cloudflare response parser test failed\\n");return 1;}
