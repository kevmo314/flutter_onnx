#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef BENCH_NEON
void onnx_pointwise_pack(const float*,const float*,float*,int,int,int,int);
int onnx_pointwise_run(const float*,const float*,float*,int,int,int,int,int,int);
#endif
static double now(void) { return clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW) * 1e-9; }
static double polynomial(double x) {
  double t = 1 / (1 + .3275911*fabs(x));
  return (((((1.061405429*t-1.453152027)*t+1.421413741)*t-.284496736)*t+.254829592)*t);
}
static void activate(float *v, int len, int fast) {
  enum { block = 128 };
  if (!fast) {
    for(int i=0;i<len;i++) {
      double x=(float)((double)v[i]/1.4142135381698608);
      float erf=x==0?x:copysign(1-polynomial(x)*exp(-x*x),x);
      v[i]=(v[i]*(1.f+erf))*.5f;
    }
    return;
  }
  for (int s=0;s<len;s+=block) {
    double x[block], e[block], p[block];
    int n=len-s<block?len-s:block;
    for(int i=0;i<n;i++) {
      x[i]=(float)((double)v[s+i]/1.4142135381698608);
      e[i]=-x[i]*x[i];p[i]=polynomial(x[i]);
    }
    vvexp(e,e,&n);
    for(int i=0;i<n;i++) {
      float erf=x[i]==0?x[i]:copysign(1-p[i]*e[i],x[i]);
      v[s+i]=(v[s+i]*(1.f+erf))*.5f;
    }
  }
}
typedef struct { int f,k,p,t,b,workers,fast,gelu;const float *w,*x;float *product,*out; } Job;
static void work(void *context, size_t index) {
  Job *q=context;
  int first=q->f*index/q->workers,end=q->f*(index+1)/q->workers;
  cblas_sgemm(CblasRowMajor,CblasTrans,CblasNoTrans,end-first,q->p,q->k,1,q->w+first,q->f,q->x,q->p,0,q->product+(size_t)first*q->p,q->p);
  for(int b=0;b<q->b;b++){
    float *dst=q->out+((size_t)b*q->f+first)*q->t;
    for(int f=first;f<end;f++){
      for(int t=0;t<q->t;t++)dst[(f-first)*q->t+t]=q->product[(size_t)f*q->p+b*q->t+t]+.34f;
    }
    if(q->gelu)activate(dst,(end-first)*q->t,q->fast);
  }
}
static void compare(int f,int k,int t,int b,int gelu) {
  int p=(b*t+3)&~3;
  float *w=malloc(f*k*4),*x=malloc(k*p*4),*out=malloc(f*t*b*4),*prod=malloc(f*p*4),*ref=malloc(f*t*b*4);
  for(int i=0;i<f*k;i++)w[i]=((float)rand()/RAND_MAX-.5f)*.1f;
  for(int i=0;i<k*p;i++)x[i]=(float)rand()/RAND_MAX-.5f;
  dispatch_queue_t pool=dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,0);
  for(int fast=0;fast<=gelu;fast++)for(int workers=1;workers<=3;workers++){
    Job j={f,k,p,t,b,workers,fast,gelu,w,x,prod,out};
    double times[7];
    for(int r=0;r<7;r++){
      double start=now();
      for(int run=0;run<20;run++) {
        if(workers==1)work(&j,0);else dispatch_apply_f(workers,pool,&j,work);
      }
      times[r]=(now()-start)*50;
    }
    for(int i=0;i<7;i++)for(int k=i+1;k<7;k++)if(times[k]<times[i]){double a=times[i];times[i]=times[k];times[k]=a;}
    if(workers==1 && fast==0)memcpy(ref,out,f*t*b*4);
    double err=0;
    for(int i=0;i<f*t*b;i++)err=fmax(err,fabs(ref[i]-out[i]));
    printf("TILE %d %d %d %d gelu=%d vector=%d threads=%d ms=%.3f error=%.9g\n",f,k,t,b,gelu,fast,workers,times[3],err);
  }
  float *nativeW=malloc(4*f*k),*xt=malloc(4*b*t*k),*yt=malloc(4*b*t*f),*bias=malloc(4*f);
  for(int j=0;j<f;j++) {bias[j]=.34f; for(int d=0;d<k;d++)nativeW[j*k+d]=w[d*f+j];}
  for(int nt=1;nt<=2;nt++){
    BNNSLayerParametersFullyConnected fc = {
      .i_desc={.layout=BNNSDataLayoutVector,.data_type=BNNSDataTypeFloat32,.size={k}},
      .o_desc={.layout=BNNSDataLayoutVector,.data_type=BNNSDataTypeFloat32,.size={f}},
      .bias={.layout=BNNSDataLayoutVector,.data_type=BNNSDataTypeFloat32,.size={f},.data=bias},
      .w_desc={.layout=BNNSDataLayoutRowMajorMatrix,.data_type=BNNSDataTypeFloat32,.size={k,f},.data=nativeW},
      .activation={.function=BNNSActivationFunctionIdentity}
    };
    BNNSFilterParameters params={.n_threads=nt};
    BNNSFilter filter=BNNSFilterCreateLayerFullyConnected(&fc,&params);
    if(!filter){puts("FC creation failed");continue;}
    double durations[7];
    for(int r=0;r<7;r++) {
      double start=now();
      for(int j=0;j<20;j++) {
        for(int z=0;z<b*t;z++)for(int d=0;d<k;d++)xt[z*k+d]=x[d*p+z];
        if(BNNSFilterApplyBatch(filter,b*t,xt,k,yt,f))abort();
        for(int v=0;v<b;v++) for(int h=0;h<f;h++)for(int z=0;z<t;z++)
          out[(v*f+h)*t+z]=yt[(v*t+z)*f+h];
        if(gelu)activate(out,b*t*f,1);
      }
      durations[r]=(now()-start)*50;
    }
    for(int i=0;i<7;i++)for(int k=i+1;k<7;k++)if(durations[k]<durations[i]){double a=durations[i];durations[i]=durations[k];durations[k]=a;}
    double err=0;for(int i=0;i<b*t*f;i++)err=fmax(err,fabs(out[i]-ref[i]));
    printf("FC %d %d %d %d gelu=%d threads=%d ms=%.3f error=%.9g\n",f,k,t,b,gelu,nt,durations[3],err);
    BNNSFilterDestroy(filter);
  }
#ifdef BENCH_NEON
  float *packed=malloc(f*(k+1)*4), *ncl=malloc(b*k*t*4);
  onnx_pointwise_pack(nativeW,bias,packed,0,f,k,f);
  for(int v=0;v<b;v++)for(int j=0;j<k;j++)memcpy(ncl+(v*k+j)*t,x+j*p+v*t,t*4);
  for(int nt=1;nt<=2;nt++){
    double start=now();
    for(int i=0;i<60;i++) {
      if(!onnx_pointwise_run(ncl,packed,out,b,k,t,f,0,nt))abort();
      if(gelu)activate(out,b*t*f,1);
    }
    double ms=(now()-start)*1000/60,err=0;
    for(int i=0;i<f*b*t;i++)err=fmax(err,fabs(ref[i]-out[i]));
    printf("NEON %d %d %d %d threads=%d ms=%.3f error=%.9g\n",f,k,t,b,nt,ms,err);
  }
  free(packed);free(ncl);
#endif
  free(nativeW);free(xt);free(yt);free(bias);
  free(w);free(x);free(out);free(prod);free(ref);
}
int main(void) {
  for(int i=0;i<2;i++)for(int t=21;t<=53;t+=16)compare(i?512:2048,i?2048:512,t,2,!i);
}
