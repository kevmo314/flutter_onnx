from pathlib import Path
from textwrap import dedent
p=Path('runtime/cpu/src/pointwise.c')
s=p.read_text()
s=s.replace('typedef struct {\n  BNNSFilter product;',dedent('''
    #include <time.h>
    #include <stdio.h>
    static uint64_t now_ns(void) {
      struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
      return (uint64_t)t.tv_sec*1000000000+t.tv_nsec;
    }
    static _Thread_local uint64_t timings[4], calls;
    static void record_stages(uint64_t start, uint64_t input, uint64_t product, uint64_t output) {
      timings[0]+=input-start; timings[1]+=product-input; timings[2]+=output-product;
      timings[3]+=now_ns()-output;
      if (++calls % 100 == 0) {
        char path[2048]; snprintf(path,sizeof(path),"%s/Documents/native-stages.log",getenv("HOME"));
        FILE *f=fopen(path,"a");
        if(f) { fprintf(f,"%llu %llu %llu %llu %llu\\n",(unsigned long long)calls,
          (unsigned long long)timings[0],(unsigned long long)timings[1],
          (unsigned long long)timings[2],(unsigned long long)timings[3]); fclose(f); }
        memset(timings,0,sizeof(timings));
      }
    }
    typedef struct {
      BNNSFilter product;'''))
s=s.replace('  const dense_filter *prepared = filter;', '  uint64_t start=now_ns();\n  const dense_filter *prepared = filter;')
s=s.replace('  const int success =','  uint64_t input=now_ns();\n  const int success =').replace('  if (success) {','  uint64_t product=now_ns();\n  if (success) {').replace('    if (gelu)\n      activate_span(y, batch * ny);','    uint64_t output=now_ns();\n    if (gelu)\n      activate_span(y, batch * ny);\n    record_stages(start,input,product,output);')
p.write_text(s)
