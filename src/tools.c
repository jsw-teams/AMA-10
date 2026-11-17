#include "tools.h"
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {const char *p;int failed;} Parser;
static void ws(Parser *p){while(*p->p&&isspace((unsigned char)*p->p))p->p++;}
static double expr(Parser *p);
static double power(Parser *p);
static double atom(Parser *p){ws(p);if(*p->p=='('){p->p++;double x=expr(p);ws(p);if(*p->p!=')')p->failed=1;else p->p++;return x;}char *end=NULL;double x=strtod(p->p,&end);if(end==p->p){p->failed=1;return 0;}p->p=end;if(!isfinite(x))p->failed=1;return x;}
static double unary(Parser *p){ws(p);if(*p->p=='+'){p->p++;return unary(p);}if(*p->p=='-'){p->p++;return -unary(p);}return power(p);}
static double power(Parser *p){double x=atom(p);ws(p);if(*p->p=='^'){p->p++;double y=unary(p);x=pow(x,y);if(!isfinite(x))p->failed=1;}return x;}
static double term(Parser *p){double x=unary(p);for(;;){ws(p);char op=*p->p;if(op!='*'&&op!='/')return x;p->p++;double y=unary(p);if(op=='/'&&y==0){p->failed=1;return 0;}x=op=='*'?x*y:x/y;if(!isfinite(x))p->failed=1;}}
static double expr(Parser *p){double x=term(p);for(;;){ws(p);char op=*p->p;if(op!='+'&&op!='-')return x;p->p++;double y=term(p);x=op=='+'?x+y:x-y;if(!isfinite(x))p->failed=1;}}
static int calculator(const char *s,char *out,size_t cap){if(!s||strlen(s)>256||!cap)return 0;Parser p={s,0};double x=expr(&p);ws(&p);if(p.failed||*p.p||!isfinite(x))return 0;int n=snprintf(out,cap,"%.12g",x);return n>0&&(size_t)n<cap;}
static int contains_fold(const char *line,const char *query){size_t q=strlen(query);if(!q)return 0;for(const char *p=line;*p;p++){size_t i=0;while(i<q&&p[i]&&tolower((unsigned char)p[i])==tolower((unsigned char)query[i]))i++;if(i==q)return 1;}return 0;}
static int text_search(const char *args,char *out,size_t cap){const char *tab=strchr(args,'\t');if(!tab||tab==args||!tab[1]||strlen(tab+1)>256)return 0;size_t path_len=(size_t)(tab-args);if(path_len>4095)return 0;char path[4096];memcpy(path,args,path_len);path[path_len]=0;FILE *f=fopen(path,"rb");if(!f)return 0;size_t used=0,line_len=0,line_cap=2048;char *line=(char*)malloc(line_cap);if(!line){fclose(f);return 0;}uint64_t line_no=0,matches=0;int ch,ok=1;
    while((ch=fgetc(f))!=EOF){if(ch=='\n'){line[line_len]=0;line_no++;if(contains_fold(line,tab+1)){matches++;if(matches<=3&&used<cap){int n=snprintf(out+used,cap-used,"line %llu: %.*s\n",(unsigned long long)line_no,(int)(line_len>512?512:line_len),line);if(n<0||(size_t)n>=cap-used){ok=0;break;}used+=(size_t)n;}}line_len=0;continue;}if(ch=='\r')continue;if(line_len+1<line_cap)line[line_len++]=(char)ch;else{ok=0;break;}}
    if(ok&&line_len){line[line_len]=0;line_no++;if(contains_fold(line,tab+1)){matches++;if(matches<=3){int n=snprintf(out+used,cap-used,"line %llu: %.*s\n",(unsigned long long)line_no,(int)(line_len>512?512:line_len),line);if(n<0||(size_t)n>=cap-used)ok=0;else used+=(size_t)n;}}}
    if(ferror(f))ok=0;
    free(line);fclose(f);
    if(!ok)return 0;
    if(matches==0){int n=snprintf(out,cap,"No matching lines.");return n>0&&(size_t)n<cap;}
    if(used<cap){int n=snprintf(out+used,cap-used,"matches=%llu",(unsigned long long)matches);return n>0&&(size_t)n<cap-used;}
    return 0;
}
int ama10_tool_execute(const char *name,const char *args,char *result,size_t cap){if(!name||!args||!result||!cap)return 0;if(!strcmp(name,"calculator"))return calculator(args,result,cap);if(!strcmp(name,"text_search"))return text_search(args,result,cap);return 0;}
int ama10_tool_dispatch(const unsigned char *text,size_t n,char *name,size_t ncap,char *result,size_t rcap){
    static const char start[]="<|tool_call|>",end[]="<|end_tool_call|>";
    if(!text||!name||!ncap||!result||!rcap)return 0;
    const unsigned char *a=NULL;
    for(size_t i=0;i+sizeof(start)-1<=n;i++)if(!memcmp(text+i,start,sizeof(start)-1)){if(a)return 0;a=text+i+sizeof(start)-1;}
    if(!a)return 0;
    size_t remain=n-(size_t)(a-text);const unsigned char *b=NULL;
    for(size_t i=0;i+sizeof(end)-1<=remain;i++)if(!memcmp(a+i,end,sizeof(end)-1)){if(b)return 0;b=a+i;}
    if(!b||b==a)return 0;
    if(memchr(a,0,(size_t)(b-a)))return 0;
    const unsigned char *tab=(const unsigned char*)memchr(a,'\t',(size_t)(b-a));
    if(!tab||tab==a||tab==b-1)return 0;
    size_t nn=(size_t)(tab-a),an=(size_t)(b-tab-1);if(nn>=ncap||an>16384)return 0;
    memcpy(name,a,nn);name[nn]=0;char *args=(char*)malloc(an+1);if(!args)return 0;
    memcpy(args,tab+1,an);args[an]=0;int ok=ama10_tool_execute(name,args,result,rcap);free(args);return ok;
}
