#include "tools.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
int main(void){char out[4096],name[64];const char call[]="prefix<|tool_call|>calculator\t-(3+1)*2 + 2^3<|end_tool_call|>suffix";
    if(!ama10_tool_execute("calculator","(2+3)*4-1.5",out,sizeof(out))||strcmp(out,"18.5")){fprintf(stderr,"calc=%s\n",out);goto fail;}
    if(!ama10_tool_execute("calculator","-2^2",out,sizeof(out))||strcmp(out,"-4")){fprintf(stderr,"unary=%s\n",out);goto fail;}
    if(ama10_tool_execute("calculator","1/0",out,sizeof(out))||ama10_tool_execute("calculator","1+system(1)",out,sizeof(out)))goto fail;
    if(!ama10_tool_dispatch((const unsigned char*)call,sizeof(call)-1,name,sizeof(name),out,sizeof(out))||strcmp(name,"calculator")||strcmp(out,"0")){fprintf(stderr,"dispatch name=%s result=%s\n",name,out);goto fail;}
    static const char unsupported[]="<|tool_call|>shell\tx<|end_tool_call|>";
    if(ama10_tool_dispatch((const unsigned char*)unsupported,sizeof(unsupported)-1,name,sizeof(name),out,sizeof(out)))goto fail;
    FILE *f=fopen("build/tool-search-fixture.txt","wb");if(!f)goto fail;fputs("alpha\nThe quick BROWN fox\nlast line",f);if(fclose(f)!=0)goto fail;
    if(!ama10_tool_execute("text_search","build/tool-search-fixture.txt\tbrown",out,sizeof(out))||!strstr(out,"line 2: The quick BROWN fox")||!strstr(out,"matches=1")){fprintf(stderr,"search=%s\n",out);goto fail;}
    puts("C calculator, bounded local search, and tool-call protocol: ok");return 0;
fail:fprintf(stderr,"C tools test failed\n");return 1;}
