#include <Accelerate/Accelerate.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void) {return clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW)*1e-9;}
static void case_(int f,int k,int t,int b) {
 const int n=b*t,cap=(n+63)&~63;
 float *w=malloc(f*k*4),*wt=malloc(f*k*4),*x=malloc(b*t*k*4),*ref=malloc(b*f*t*4);
 float *a=malloc(k*cap*4),*c=malloc(f*cap*4),*y=malloc(b*f*t*4);
 for(int i=0;i<f*k;i++)w[i]=(float)rand()/RAND_MAX-.5f;
 for(int i=0;i<b*t*k;i++)x[i]=(float)rand()/RAND_MAX-.5f;
 for(int i=0;i<f;i++)for(int j=0;j<k;j++)wt[j*f+i]=w[i*k+j];
 double start=now();
 for(int test=0;test<20;test++)for(int v=0;v<b;v++)
   cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,f,t,k,1,w,k,x+v*t*k,t,0,ref+v*f*t,t);
 double base=(now()-start)/20, best=1e10;int mode_best=0,align_best=0;
 printf("BASE %d %d %d %d %.3f ms\n",f,k,t,b,base*1000);
 for(int mode=0;mode<4;mode++)for(int align=1;align<=64;align*=2) {
  if(align>1 && ((n+align-1)&~(align-1))==((n+align/2-1)&~(align/2-1)))continue;
  int p=(n+align-1)&~(align-1);
  memset(a,0,k*p*4);start=now();
  for(int trial=0;trial<20;trial++) {
   for(int j=0;j<k;j++)for(int v=0;v<b;v++)
     memcpy(a+j*p+v*t,x+(v*k+j)*t,t*4);
   if(mode<2)cblas_sgemm(CblasRowMajor,mode?CblasTrans:CblasNoTrans,CblasNoTrans,
       f,p,k,1,mode?wt:w,mode?f:k,a,p,0,c,p);
   else cblas_sgemm(CblasRowMajor,CblasTrans,mode==2?CblasNoTrans:CblasTrans,
       p,f,k,1,a,p,mode==2?wt:w,mode==2?f:k,0,c,f);
   for(int j=0;j<f;j++)for(int v=0;v<b;v++) {
     if(mode<2)memcpy(y+(v*f+j)*t,c+j*p+v*t,t*4);
     else for(int s=0;s<t;s++)y[(v*f+j)*t+s]=c[(v*t+s)*f+j];
   }
  }
  double ms=(now()-start)/20,diff=0;
  for(int j=0;j<b*f*t;j++)diff=fmax(diff,fabs(ref[j]-y[j]));
  if(ms<best){best=ms;mode_best=mode;align_best=align;}
  printf("LAYOUT %d %d %d %d %d %d %.3f ms diff=%g\n",f,k,t,b,mode,align,ms*1000,diff);
 }
 printf("BEST %d %d %d %d mode=%d align=%d time=%.3f speedup=%.2f\n",f,k,t,b,mode_best,align_best,best*1000,base/best);
 free(w);free(wt);free(x);free(a);free(c);free(y);free(ref);
}
int main(void){for(int i=0;i<2;i++){int f=i?512:2048,k=i?2048:512;case_(f,k,21,2);case_(f,k,52,2);case_(f,k,26,2);case_(f,k,21,1);}return 0;}
