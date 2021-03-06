#include<vector>
#include<iostream>
using namespace std;
#define USE_OMP 0
#define OMP_THREAD 8


// #if __AVX__
// #include "avx_activation.h"
// #include "avx_usability.h"
// #endif

#if __SSE2__
#include <emmintrin.h>
#endif
#if __AVX__
#include <immintrin.h>
#endif


using namespace std;

    void convolutionTransformKernel(float* kernel, const int &kernelW, const int &kernelH, 
            float * &dest, const int &inChannel, const int &outChannel){

            int kernelSize = kernelW*kernelH;
            int ccOutChannel=0;
            int ccRemainOutChannel = 0;
            int Stride=0;

            ccOutChannel = outChannel >> 2;//整除的channel
            ccRemainOutChannel = outChannel<< 2;//剩余的甩尾channel

            for(int cc=0; cc<ccOutChannel; cc++){
                int c = cc<<2;
                const float *k0 = kernel + c * inChannel * kernelSize;
                const float *k1 = kernel + (c + 1) * inChannel *kernelSize;
                const float *k2 = kernel + (c + 2) * inChannel * kernelSize;
                const float *k3 = kernel + (c + 3) * inChannel * kernelSize;

                Stride = 4 * kernelSize * inChannel;
                float* destptr = dest + (c / 4)* Stride;
                for(int i=0; i< inChannel* kernelSize; i++){
                    destptr[0] = k0[0];
                    destptr[1] = k1[0];
                    destptr[2] = k2[0];
                    destptr[3] = k3[0];

                    destptr +=4;

                    k0 +=1;
                    k1 +=1;
                    k2 +=1;
                    k3 +=1;
                }
            }
            //如果channel不能被4整除，需要处理剩余部分
            for(int cc = ccRemainOutChannel; cc < outChannel; cc++){
                int c =cc;
                const float *k0 = kernel + c * inChannel * kernelSize;
                Stride = 4 * kernelSize * inChannel;
                float * destptr = dest + (c / 4 + c % 4)* Stride;
                for(int i = 0; i < inChannel * kernelSize; i++){
                    destptr[0] = k0[0];
                    destptr +=1;

                    k0 +=1;
                }
            }


    }
    

    void convolutionIm2colSgemm(float *const &src, const int &inWidth, const int &inHeight,
            const int &inChannel, float *const &kernel, float *const kernel_im2col_pack,
            const int &kernelW, const int &kernelH, float * &dest, const int &outWidth, const int &outHeight, const int &outChannel,
            const int &StrideH, const int &StrideW){
                // 1. im2col
        // src_im2col : width=outWidth * outHeight, height=kernelH * kernelW * inChannel
                float *src_im2col = new float [outWidth * outHeight *kernelW * kernelH* inChannel];
                const int Stride = kernelW * kernelH *outHeight* outWidth;

                const int outSize = outWidth*outHeight;
                const int kernelSize = kernelH * kernelW;

#if USE_OMP
    #pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for(int cc = 0; cc < inChannel; cc++){
            const float *src0 = src + cc *inHeight * inWidth;
            int dst_idx = Stride*cc;
            for(int i = 0; i <kernelH; i++){
                for(int j = 0; j <kernelW; j++){
                    for(int x = 0; x <outHeight; x++){
                        for(int y = 0; y <outWidth; y++){
                            int row = x *StrideH +i;
                            int col = y *StrideW +j;
                            int ori_idx = row *inWidth +col;
                            src_im2col[dst_idx] = src0[ori_idx];
                            dst_idx++;
                        }

                    }
                }
            }

        }
        //pack 8x8
        const int packChannel = outSize / 8 + outSize% 8;
        const int packHeight = inChannel;
        const int packWidth = 8* kernelSize;

        int kernelPackChannel = outChannel/4 + outChannel % 4;
        const int kernelPackHeight =inChannel;
        const int kernelPackWidth = 4 * kernelSize;

        float *src_im2col_pack = new float[packWidth*packHeight* packChannel];
        
        int colCount = outSize>> 3;
        int remainColCount = colCount<<3;

#if USE_OMP
    #pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i <colCount; i++){
            int newi = i<<3;
            const float *src0 = src_im2col;

            src0 +=newi;

            float *packptr = src_im2col_pack + i *packHeight* packWidth;

            for(int j = 0; j < inChannel * kernelSize; j ++){

#if __AVX__
                _mm256_storeu_ps(packptr, _mm256_loadu_ps(src0));
#else                                
                packptr[0] = src0[0];
                packptr[1] = src0[1];
                packptr[2] = src0[2];
                packptr[3] = src0[3];
                packptr[4] = src0[4];
                packptr[5] = src0[5];
                packptr[6] = src0[6];
                packptr[7] = src0[7];
#endif
                packptr +=8;
                src0 += outSize;
            }
        }

        //pack tail
#if USE_OMP
    #pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = remainColCount; i< outSize; i++){
            const float *src0 = src_im2col;
            src0 +=i;
            float *packptr = src_im2col_pack +(i / 8 + i % 8)*  packHeight * packWidth;

            for(int j = 0; j < inChannel * kernelSize; j ++){
                packptr[0] = src0[0];

                packptr += 1;
                src0 += outSize;
            }
        }
        //pack end

        //sgemm

        int N = outHeight*outWidth;
        int K = kernelSize * inChannel;

        int ccOutChannel = outChannel >>2;
        int ccRemainOutChannel = ccOutChannel << 2;

#if USE_OMP
    #pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for(int cc = 0; cc < ccOutChannel; cc++){
            int c = cc<<2;
            float *destptr0 = dest + c* outSize;
            float *destptr1 = dest + (c + 1) * outSize;
            float *destptr2 = dest + (c + 2) * outSize;
            float *destptr3 = dest + (c + 3) * outSize;

            int i = 0;
            const float zeros[4] = {0.f, 0.f, 0.f, 0.f};
            const float *biasptr = zeros;

            for(; i+7 < N; i = i + 8){
                const float *ptrB = src_im2col_pack + (i / 8) *  packHeight * packWidth;
                const float *ptrA = kernel_im2col_pack + (c / 4)* kernelPackHeight * kernelPackWidth;

#if __AVX__
                __m256 _sum0 = _mm256_broadcast_ss(biasptr);
                __m256 _sum1 = _mm256_broadcast_ss(biasptr +1);
                __m256 _sum2 = _mm256_broadcast_ss(biasptr +2);
                __m256 _sum3 = _mm256_broadcast_ss(biasptr +3);

                int m = 0;
                for (; m +3 < K; m +=4){

                    //k0
                    __m256 _va0 = _mm256_broadcast_ss(ptrA);
                    __m256 _va1 = _mm256_broadcast_ss(ptrA + 1);
                    __m256 _va2 = _mm256_broadcast_ss(ptrA + 2);
                    __m256 _va3 = _mm256_broadcast_ss(ptrA + 3);
                    __m256 _vb0 = _mm256_broadcast_ss(ptrB);
                    __m256 _vb1 = _mm256_broadcast_ss(ptrB + 8);
                    __m256 _vb2 = _mm256_broadcast_ss(ptrB + 16);
                    __m256 _vb3 = _mm256_broadcast_ss(ptrB + 24);
                    _sum0 = _mm256_fmadd_ps(_vb0, _va0, _sum0);
                    _sum1 = _mm256_fmadd_ps(_vb1, _va1, _sum1);
                    _sum2 = _mm256_fmadd_ps(_vb2, _va2, _sum2);
                    _sum3 = _mm256_fmadd_ps(_vb3, _va3, _sum3);
                    ptrA += 4;

                    // k1
                    _va0 = _mm256_broadcast_ss(ptrA);
                    _va1 = _mm256_broadcast_ss(ptrA + 1);
                    _va2 = _mm256_broadcast_ss(ptrA + 2);
                    _va3 = _mm256_broadcast_ss(ptrA + 3);
                    _sum0 = _mm256_fmadd_ps(_vb1, _va0, _sum0); // sum0 += (a10-a17) * k01
                    _sum1 = _mm256_fmadd_ps(_vb1, _va1, _sum1); // sum1 += (a10-a17) * k11
                    _sum2 = _mm256_fmadd_ps(_vb1, _va2, _sum2); // sum2 += (a10-a17) * k21
                    _sum3 = _mm256_fmadd_ps(_vb1, _va3, _sum3); // sum3 += (a10-a17) * k31

                    ptrA += 4;

                    // k2
                    _va0 = _mm256_broadcast_ss(ptrA);
                    _va1 = _mm256_broadcast_ss(ptrA + 1);
                    _va2 = _mm256_broadcast_ss(ptrA + 2);
                    _va3 = _mm256_broadcast_ss(ptrA + 3);
                    _sum0 = _mm256_fmadd_ps(_vb2, _va0, _sum0); // sum0 += (a20-a27) * k02
                    _sum1 = _mm256_fmadd_ps(_vb2, _va1, _sum1); // sum1 += (a20-a27) * k12
                    _sum2 = _mm256_fmadd_ps(_vb2, _va2, _sum2); // sum2 += (a20-a27) * k22
                    _sum3 = _mm256_fmadd_ps(_vb2, _va3, _sum3); // sum3 += (a20-a27) * k32

                    ptrA += 4;

                    // k3
                    _va0 = _mm256_broadcast_ss(ptrA);
                    _va1 = _mm256_broadcast_ss(ptrA + 1);
                    _va2 = _mm256_broadcast_ss(ptrA + 2);
                    _va3 = _mm256_broadcast_ss(ptrA + 3);
                    _sum0 = _mm256_fmadd_ps(_vb3, _va0, _sum0); // sum0 += (a30-a37) * k03
                    _sum1 = _mm256_fmadd_ps(_vb3, _va1, _sum1); // sum1 += (a30-a37) * k13
                    _sum2 = _mm256_fmadd_ps(_vb3, _va2, _sum2); // sum2 += (a30-a37) * k23
                    _sum3 = _mm256_fmadd_ps(_vb3, _va3, _sum3); // sum3 += (a30-a37) * k33

                    ptrA += 4;
                    ptrB += 32;
                }

                for(; m<K; m++){
                    // k0
                    __m256 _va0 = _mm256_broadcast_ss(ptrA);
                    __m256 _va1 = _mm256_broadcast_ss(ptrA + 1);
                    __m256 _va2 = _mm256_broadcast_ss(ptrA + 2);
                    __m256 _va3 = _mm256_broadcast_ss(ptrA + 3);
                    __m256 _vb0 = _mm256_loadu_ps(ptrB);
                    _sum0 = _mm256_fmadd_ps(_vb0, _va0, _sum0); // sum0 = (a00-a07) * k00
                    _sum1 = _mm256_fmadd_ps(_vb0, _va1, _sum1); // sum1 = (a00-a07) * k10
                    _sum2 = _mm256_fmadd_ps(_vb0, _va2, _sum2); // sum2 = (a00-a07) * k20
                    _sum3 = _mm256_fmadd_ps(_vb0, _va3, _sum3); // sum3 = (a00-a07) * k30

                    ptrA += 4;
                    ptrB += 8;
                }

                _mm256_storeu_ps(destptr0, _sum0);
                _mm256_storeu_ps(destptr1, _sum1);
                _mm256_storeu_ps(destptr2, _sum2);
                _mm256_storeu_ps(destptr3, _sum3);
#else                               
                float sum0[8]= {0};//pack之后的每一列
                float sum1[8]= {0};
                float sum2[8]= {0};
                float sum3[8]= {0};

                int j=0;
                // K = kernelSize * inChannel
                // 同时计算4行，同时在每一列计算8个输出
                for(; j +7< K; j = j + 8){
                    for(int n = 0; n < 8; n++){
                        sum0[n] += ptrA[0] * ptrB[n];
                        sum1[n] += ptrA[1] * ptrB[n];
                        sum2[n] += ptrA[2] * ptrB[n];
                        sum3[n] += ptrA[3] * ptrB[n];
                        ptrA += 4;

                        sum0[n] += ptrA[0] * ptrB[n + 8];
                        sum1[n] += ptrA[1] * ptrB[n + 8];
                        sum2[n] += ptrA[2] * ptrB[n + 8];
                        sum3[n] += ptrA[3] * ptrB[n + 8];
                        ptrA += 4;

                        sum0[n] += ptrA[0] * ptrB[n + 16];
                        sum1[n] += ptrA[1] * ptrB[n + 16];
                        sum2[n] += ptrA[2] * ptrB[n + 16];
                        sum3[n] += ptrA[3] * ptrB[n + 16];
                        ptrA += 4;

                        sum0[n] += ptrA[0] * ptrB[n + 24];
                        sum1[n] += ptrA[1] * ptrB[n + 24];
                        sum2[n] += ptrA[2] * ptrB[n + 24];
                        sum3[n] += ptrA[3] * ptrB[n + 24];
                        ptrA += 4;

                        sum0[n] += ptrA[0] * ptrB[n + 32];
                        sum1[n] += ptrA[1] * ptrB[n + 32];
                        sum2[n] += ptrA[2] * ptrB[n + 32];
                        sum3[n] += ptrA[3] * ptrB[n + 32];
                        ptrA += 4;

                        sum0[n] += ptrA[0] * ptrB[n + 40];
                        sum1[n] += ptrA[1] * ptrB[n + 40];
                        sum2[n] += ptrA[2] * ptrB[n + 40];
                        sum3[n] += ptrA[3] * ptrB[n + 40];
                        ptrA += 4;

                        sum0[n] += ptrA[0] * ptrB[n + 48];
                        sum1[n] += ptrA[1] * ptrB[n + 48];
                        sum2[n] += ptrA[2] * ptrB[n + 48];
                        sum3[n] += ptrA[3] * ptrB[n + 48];
                        ptrA += 4;

                        sum0[n] += ptrA[0] * ptrB[n + 56];
                        sum1[n] += ptrA[1] * ptrB[n + 56];
                        sum2[n] += ptrA[2] * ptrB[n + 56];
                        sum3[n] += ptrA[3] * ptrB[n + 56];
                        ptrA -= 28;

                    }

                    ptrA +=32;
                    ptrB +=64;
                }
                // K = kernelSize * inChannel * 4
                // 如果是pack4x4那么末尾一定是4的倍数
                for(; j < K; j++){
                    for(int n = 0; n < 8 ; n++){
                        sum0[n] += ptrA[0] * ptrB[n];
                        sum1[n] += ptrA[1] * ptrB[n];
                        sum2[n] += ptrA[2] * ptrB[n];
                        sum3[n] += ptrA[3] * ptrB[n];
                    }
                    ptrA += 4;
                    ptrB += 8;
                }
                for(int n = 0; n < 8; n++){
                    destptr0[n] = sum0[n];
                    destptr1[n] = sum1[n];
                    destptr2[n] = sum2[n];
                    destptr3[n] = sum3[n];
                }
#endif
                destptr0 +=8;
                destptr1 +=8;
                destptr2 +=8;
                destptr3 +=8;

            }
            // N = outHeight*outWidth
            // 拖尾部分，在列方向上只能逐个计算
            for(; i<N;i++){
                const float *ptrB = src_im2col_pack + (i / 8 + i % 8) *  packHeight * packWidth;
                const float *ptrA = kernel_im2col_pack +(c / 4)*kernelPackHeight* kernelPackWidth;

                float sum0 = 0;
                float sum1 = 0;
                float sum2 = 0;
                float sum3 = 0;
                // K = kernelSize * inChannel * 4
                for(int j = 0; j<K; j++){
                    sum0 +=ptrA[0] * ptrB[0];
                    sum1 +=ptrA[1] * ptrB[0];
                    sum2 +=ptrA[2] * ptrB[0];
                    sum3 +=ptrA[3] * ptrB[0];

                    ptrA +=4;
                    ptrB +=1;
                }

                destptr0[0] = sum0;
                destptr1[0] = sum1;
                destptr2[0] = sum2;
                destptr3[0] = sum3;

                destptr0++;
                destptr1++;
                destptr2++;
                destptr3++;

            }
        }


#if USE_OMP
    #pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for(int cc = ccRemainOutChannel; cc < outChannel; cc++){
            int c = cc;
            float *destptr0 = dest + c * outSize;
            int i =0;
            for(; i+7 < N; i = i + 8){
                const float *ptrB = src_im2col_pack + (i / 8) *  packHeight * packWidth;
                const float *ptrA = kernel_im2col_pack + (c / 4)* kernelPackHeight * kernelPackWidth;

                float sum[8] = {0};
                int j = 0;
                for(;j +7< K;j = j + 8){
                    for(int n = 0; n < 8; n++){
                        sum[n] += ptrA[0] * ptrB[n];
                        sum[n] += ptrA[1] * ptrB[n +8];
                        sum[n] += ptrA[2] * ptrB[n + 16];
                        sum[n] += ptrA[3] * ptrB[n + 24];
                        sum[n] += ptrA[4] * ptrB[n + 32];
                        sum[n] += ptrA[5] * ptrB[n + 40];
                        sum[n] += ptrA[6] * ptrB[n + 48];
                        sum[n] += ptrA[7] * ptrB[n + 56];
                    }
                    ptrA += 8;
                    ptrB += 64;
                }
                for(; j< K; j++){
                    for(int n=0; n<8; n++){
                        sum[n] += ptrA[0]* ptrB[n];
                    }
                    ptrA += 1;
                    ptrB += 8;
                }

                for(int n=0; n < 8; n++){
                    destptr0[n] = sum[n];
                }
                destptr0 +=8;
            }

            for(; i< N; i++){
                const float *ptrB = src_im2col_pack + (i / 8 + i % 8) *  packHeight * packWidth;
                const float *ptrA = kernel_im2col_pack + (c / 4)* kernelPackHeight * kernelPackWidth;
                int j = 0;

                float sum =0;
                for(; j<K; j++){
                    sum +=ptrA[0]* ptrB[0];
                    ptrA +=1;
                    ptrB +=1;
                }
                destptr0[0] =sum;
                destptr0++;
            }
        }

        delete[] src_im2col;
        delete[] src_im2col_pack;    
    
    
    
    }


    


        