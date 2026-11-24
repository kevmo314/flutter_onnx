#include <Accelerate/Accelerate.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void bench(int rows,int in,int out) {
  int maxrows=(rows+31)&~31;
  float *x=calloc((size_t)maxrows*in,4),*w=malloc((size_t)out*in*4),*wt=malloc((size_t)out*in*4);
  float *y=malloc((size_t)maxrows*out*4),*base=malloc((size_t)maxrows*out*4);
  for(int i=0;i<rows*in;i++)x[i]=((float)rand()/RAND_MAX-.5f)*.13f;
  for(int i=0;i<out;i++)for(int j=0;j<in;j++)wt[(size_t)j*out+i]=w[(size_t)i*in+j]=((float)rand()/RAND_MAX-.5f)*.09f;
  BNNSLayerParametersFullyConnected layer={
    .i_desc={.layout=BNNSDataLayoutVector,.data_type=BNNSDataTypeFloat32,.size={in}},
    .o_desc={.layout=BNNSDataLayoutVector,.data_type=BNNSDataTypeFloat32,.size={out}},
    .w_desc={.layout=BNNSDataLayoutRowMajorMatrix,.data_type=BNNSDataTypeFloat32,.size={in,out},.data=w},
    .activation={.function=BNNSActivationFunctionIdentity}};
  BNNSFilterParameters params={.n_threads=2};
  BNNSFilter fc=BNNSFilterCreateLayerFullyConnected(&layer,&params);
  if(!fc || BNNSFilterApplyBatch(fc,rows,x,in,base,out))abort();
  const int pads[]={1,8,16,32};
  for(int trial=0;trial<3;trial++) {
    for(int k=0;k<12;k++) {
      int variant=(trial*5+k)%12, mode=variant/4, pad=pads[variant%4], n=(rows+pad-1)/pad*pad;
      uint64_t t=clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW);
      for(int i=0;i<12;i++) {
        if(mode==0) { if(BNNSFilterApplyBatch(fc,n,x,in,y,out))abort(); }
        else cblas_sgemm(CblasRowMajor,CblasNoTrans,mode==1?CblasTrans:CblasNoTrans,n,out,in,
                  1,x,in,mode==1?w:wt,mode==1?in:out,0,y,out);
      }
      double ms=(clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW)-t)/12e6, err=0;
      for(int i=0;i<rows*out;i++)err=fmax(err,fabs(y[i]-base[i]));
      printf("K %d %d %d %d %d %d %.4f %.9g\n",rows,in,out,mode,pad,trial,ms,err);
    }
  }
  BNNSFilterDestroy(fc);free(x);free(w);free(wt);free(y);free(base);
}
int main(void){int sizes[]={21,42,74,106};for(int j=0;j<4;j++){bench(sizes[j],512,2048);bench(sizes[j],2048,512);}return 0;}
