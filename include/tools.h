#ifndef AMA10_TOOLS_H
#define AMA10_TOOLS_H
#include <stddef.h>

/* Explicit block protocol: <|tool_call|>name<TAB>arguments<|end_tool_call|> */
int ama10_tool_execute(const char *name,const char *arguments,char *result,size_t capacity);
int ama10_tool_dispatch(const unsigned char *assistant_text,size_t text_bytes,
                        char *tool_name,size_t name_capacity,
                        char *result,size_t result_capacity);
#endif
