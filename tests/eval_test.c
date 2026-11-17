#include "eval.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

int main(void){
    AMA10Tokenizer tokenizer={0};AMA10GenerationMetrics metrics;
    static const unsigned char ascii[]="ABCDABCD";
    const uint32_t ids[]={65,66,67,68,65,66,67,68};
    if(!ama10_tokenizer_train(&tokenizer,ascii,sizeof(ascii)-1,256)||
       !ama10_generation_metrics(&tokenizer,ids,8,ascii,sizeof(ascii)-1,&metrics))return 1;
    if(metrics.distinct_token_ratio!=0.5||metrics.repeated_fourgram_ratio!=0.2||
       metrics.codepoints!=8||metrics.invalid_utf8_bytes||metrics.latin!=8)return 2;
    static const unsigned char scripts[]={
        0xe4,0xb8,0xad, /* Han */
        0xed,0x95,0x9c, /* Hangul */
        0xd8,0xa7,       /* Arabic */
        0xe0,0xa4,0x85, /* Devanagari */
        0xd0,0x96,       /* Cyrillic */
        0xe0,0xa6,0x85  /* Bengali */
    };
    if(!ama10_generation_metrics(&tokenizer,ids,8,scripts,sizeof(scripts),&metrics)||
       metrics.cjk!=1||metrics.hangul!=1||metrics.arabic!=1||
       metrics.devanagari!=1||metrics.cyrillic!=1||metrics.bengali!=1||metrics.codepoints!=6)return 3;
    static const unsigned char invalid[]={0xf0,0x28,0x8c,0x28};
    if(!ama10_generation_metrics(&tokenizer,ids,8,invalid,sizeof(invalid),&metrics)||
       metrics.invalid_utf8_bytes!=2)return 4;
    {
        static const unsigned char reference[]="The quick, brown fox!";
        static const unsigned char hypothesis[]="the quick brown fix.";
        double wer=0.0,cer=0.0;uint64_t word_errors=0,words=0,char_errors=0,characters=0;
        if(!ama10_text_error_rates(reference,sizeof(reference)-1,hypothesis,
             sizeof(hypothesis)-1,&wer,&cer,&word_errors,&words,&char_errors,&characters)||
           wer!=0.25||word_errors!=1||words!=4||char_errors!=1||characters!=19||
           fabs(cer-1.0/19.0)>1e-12)return 5;
        static const unsigned char unicode_reference[]={0xe7,0x8c,0xab,0xe7,0x8a,0xac};
        static const unsigned char unicode_hypothesis[]={0xe7,0x8c,0xab,0xe9,0xb3,0xa5};
        if(!ama10_text_error_rates(unicode_reference,sizeof(unicode_reference),
             unicode_hypothesis,sizeof(unicode_hypothesis),&wer,&cer,NULL,NULL,NULL,NULL)||
           wer!=1.0||cer!=0.5)return 6;
    }
    ama10_tokenizer_free(&tokenizer);
    puts("generation repetition, Unicode script blocks, and normalized WER/CER metrics: ok");return 0;
}
