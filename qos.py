from pathlib import Path
p=Path("../runtime/cpu/src/pointwise.c");s=p.read_text();s=s.replace('enum { filters_per_tile', '''#include <pthread.h>
#include <pthread/qos.h>
#include <stdio.h>
#include <stdatomic.h>
#define onnx_fc_run onnx_fc_run_base
#define onnx_pointwise_run onnx_pointwise_run_base
enum { filters_per_tile''')
p.write_text(s+'''
#undef onnx_fc_run
#undef onnx_pointwise_run
static _Atomic int priority_enabled, logged;
EXPORT void onnx_test_priority(int enabled) { atomic_store(&priority_enabled,enabled); }
static pthread_override_t boost(void) {
  if(!atomic_exchange(&logged,1)) fprintf(stderr,"Speech thread QoS %d, override %d\\n",(int)qos_class_self(),atomic_load(&priority_enabled));
  if(!atomic_load(&priority_enabled) || qos_class_self() >= QOS_CLASS_USER_INITIATED) return NULL;
  return pthread_override_qos_class_start_np(pthread_self(),QOS_CLASS_USER_INITIATED,0);
}
EXPORT int onnx_fc_run(void *f,const float *x,float *y,int b,int c,int w,int n) {
  pthread_override_t scope=boost(); int res=onnx_fc_run_base(f,x,y,b,c,w,n);
  if(scope)pthread_override_qos_class_end_np(scope);return res;
}
EXPORT int onnx_pointwise_run(const float *x,const float *w,float *y,int b,int c,int t,int f,int g,int n) {
  pthread_override_t scope=boost(); int res=onnx_pointwise_run_base(x,w,y,b,c,t,f,g,n);
  if(scope)pthread_override_qos_class_end_np(scope);return res;
}
''')
p=Path('../runtime/cpu/lib/flutter_onnx_cpu.dart');s=p.read_text();s+="\n@Native<Void Function(Int32)>(symbol:'onnx_test_priority')\nexternal void testSpeechPriority(int enabled);\n";p.write_text(s)
p=Path('packages/on_device_tts/lib/src/onnx/graph_worker.dart');s=p.read_text();anchor='  final graph = OnnxSession.fromBytes';assert s.count(anchor)==1
p.write_text(s.replace(anchor,"  testSpeechPriority(File('${File(path).parent.parent.parent.path}/qos-mode').readAsStringSync().trim() == 'on' ? 1 : 0);\n"+anchor))
