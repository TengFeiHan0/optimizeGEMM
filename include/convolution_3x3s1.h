#include <vector>
#include <iostream>
#define USE_OMP 1
#define OMP_THREAD 8
using namespace std;

void conv3x3s1(float *const &src, const int &inWidth, const int &inHeight,  const int &inChannel,
        float *const &kernel, float * &dest, const int &outWidth, const int &outHeight, const int &outChannel){
            int ccOutChannel = outChannel>>1;
            int ccRemainOutChannel = ccOutChannel << 1;

            const int inSize = inHeight * inWidth;
            const int outSize = outHeight * outWidth;
#if USE_OMP
    #pragma omp parallel for num_threads(OMP_THREAD)
#endif
    for(int cc = 0; cc < ccOutChannel; cc++){
        int c = cc<<1;
        float *dest0 = dest + c * outSize;
        float *dest1 =  dest + (c + 1) * outSize;

        //two output rely on two kernel
        float *k0 = kernel + c * inChannel * 3 * 3;
        float *k1 = kernel + (c + 1) * inChannel * 3 * 3;

        for(int q = 0; q < inChannel; q++){
            float * destptr0 = dest0;
            float * destptr1 = dest1;
            float * destptr0_next = destptr0 + outWidth;
            float * destptr1_next = destptr1 + outWidth;

            const float* src0 = src + q * inSize;

            //deal four lines and get two outputs in a feature map
            const float* r0 = src0;
            const float* r1 = src0 + inWidth;
            const float* r2 = src0 + inWidth * 2;
            const float* r3 = src0 + inWidth * 3;

            int i = 0;
            for(; i+1 < outHeight; i += 2){
                int remain = outWidth;

                for(; remain > 0; remain--){
                    float sum0 = 0.f;
                    float sum1 = 0.f;
                    float sum0next = 0.f;
                    float sum1next = 0.f;

                    //conv output1->chanel q output1 
                    sum0 += r0[0] * k0[0];
                    sum0 += r0[1] * k0[1];
                    sum0 += r0[2] * k0[2];
                    sum0 += r1[0] * k0[3];
                    sum0 += r1[1] * k0[4];
                    sum0 += r1[2] * k0[5];
                    sum0 += r2[0] * k0[6];
                    sum0 += r2[1] * k0[7];
                    sum0 += r2[2] * k0[8];

                    //conv output1->channel q output2
                    sum1 += r0[0] * k1[0];
                    sum1 += r0[1] * k1[1];
                    sum1 += r0[2] * k1[2];
                    sum1 += r1[0] * k1[3];
                    sum1 += r1[1] * k1[4];
                    sum1 += r1[2] * k1[5];
                    sum1 += r2[0] * k1[6];
                    sum1 += r2[1] * k1[7];
                    sum1 += r2[2] * k1[8];

                    //conv output2->channel q output1
                    sum0next += r1[0] * k0[0];
                    sum0next += r1[1] * k0[1];
                    sum0next += r1[2] * k0[2];
                    sum0next += r2[0] * k0[3];
                    sum0next += r2[1] * k0[4];
                    sum0next += r2[2] * k0[5];
                    sum0next += r3[0] * k0[6];
                    sum0next += r3[1] * k0[7];
                    sum0next += r3[2] * k0[8];

                    //conv output2->channel q output2
                    sum1next += r1[0] * k1[0];
                    sum1next += r1[1] * k1[1];
                    sum1next += r1[2] * k1[2];
                    sum1next += r2[0] * k1[3];
                    sum1next += r2[1] * k1[4];
                    sum1next += r2[2] * k1[5];
                    sum1next += r3[0] * k1[6];
                    sum1next += r3[1] * k1[7];
                    sum1next += r3[2] * k1[8];

                    //sum to dest
                    *destptr0 += sum0;
                    *destptr1 += sum1;
                    *destptr0_next += sum0next;
                    *destptr1_next += sum1next;

                    r0++;
                    r1++;
                    r2++;
                    r3++;
                    destptr0++;
                    destptr1++;
                    destptr0_next++;
                    destptr1_next++;

                }

                r0 += 2 +inWidth;
                r1 += 2 +inWidth;
                r2 += 2 +inWidth;
                r3 += 2 +inWidth;

                destptr0 +=outWidth;
                destptr1 +=outWidth;
                destptr0_next +=outWidth;
                destptr1_next +=outWidth;
            }

            for(; i< outHeight; i++) {
                int remain = outWidth;

                for(; remain > 0; remain--){
                    float sum0 = 0.f;
                    float sum1 = 0.f;

                    //conv output1->chanel q output1
                    sum0 += r0[0] * k0[0];
                    sum0 += r0[1] * k0[1];
                    sum0 += r0[2] * k0[2];
                    sum0 += r1[0] * k0[3];
                    sum0 += r1[1] * k0[4];
                    sum0 += r1[2] * k0[5];
                    sum0 += r2[0] * k0[6];
                    sum0 += r2[1] * k0[7];
                    sum0 += r2[2] * k0[8];

                    //conv output2->channel q output1
                    sum1 += r0[0] * k1[0];
                    sum1 += r0[1] * k1[1];
                    sum1 += r0[2] * k1[2];
                    sum1 += r1[0] * k1[3];
                    sum1 += r1[1] * k1[4];
                    sum1 += r1[2] * k1[5];
                    sum1 += r2[0] * k1[6];
                    sum1 += r2[1] * k1[7];
                    sum1 += r2[2] * k1[8];

                    //sum to dest
                    *destptr0 += sum0;
                    *destptr1 += sum1;

                    r0++;
                    r1++;
                    r2++;
                    destptr0++;
                    destptr1++;

                }

                r0 +=2;
                r1 +=2;
                r2 +=2;
            }
            k0 +=9;
            k1 +=9;
        }


    }
#if USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
    for(int cc = 0; cc < ccOutChannel; cc++){
        int c = cc;
        float *dest0 = dest + c * outSize;
        for(int j = 0; j < outSize; j++) dest0[j] = 0.f;
        const float* kernel0 = kernel + c * inChannel * 3 * 3;
        
        for(int q = 0; q < inChannel; q++){
            float * destptr0 = dest0;
            float * destptr1 = dest0 +outWidth;
            const float* src0 = src + q * inSize;
            //deal four lines and get two outputs in a feature map
            const float* r0 = src0;
            const float* r1 = src0 + inWidth;
            const float* r2 = src0 + inWidth * 2;
            const float* r3 = src0 + inWidth * 3;

            const float* k0 = kernel0;
            const float* k1 = kernel +3;
            const float* k2 = kernel +6;

            int i=0;
            for(; i+1< outHeight; i += 2){
                int remain = outWidth;

                for(; remain > 0; remain--){
                    float sum0 = 0;
                    float sum1 = 0;

                    //conv output1->chanel q output1 
                    sum0 += r0[0] * k0[0];
                    sum0 += r0[1] * k0[1];
                    sum0 += r0[2] * k0[2];
                    sum0 += r1[0] * k1[0];
                    sum0 += r1[1] * k1[1];
                    sum0 += r1[2] * k1[2];
                    sum0 += r2[0] * k2[0];
                    sum0 += r2[1] * k2[1];
                    sum0 += r2[2] * k2[2];

                    //conv output1->channel q output2
                    sum1 += r1[0] * k0[0];
                    sum1 += r1[1] * k0[1];
                    sum1 += r1[2] * k0[2];
                    sum1 += r2[0] * k1[0];
                    sum1 += r2[1] * k1[1];
                    sum1 += r2[2] * k1[2];
                    sum1 += r3[0] * k2[0];
                    sum1 += r3[1] * k2[1];
                    sum1 += r3[2] * k2[2];

                    *destptr0 += sum0;
                    *destptr1 += sum1;

                    r0++;
                    r1++;
                    r2++;
                    r3++;
                    destptr0++;
                    destptr1++;
                }

                r0 +=2 + inWidth;
                r1 +=2 + inWidth;
                r2 +=2 + inWidth;
                r3 +=2 + inWidth;

                destptr0 += outWidth;
                destptr1 += outWidth;
            }

            for(; i< outHeight; i++){
                int remain = outWidth;
                for(; remain > 0; remain--){
                     float sum0 = 0;

                    sum0 += r0[0] * k0[0];
                    sum0 += r0[1] * k0[1];
                    sum0 += r0[2] * k0[2];
                    sum0 += r1[0] * k1[0];
                    sum0 += r1[1] * k1[1];
                    sum0 += r1[2] * k1[2];
                    sum0 += r2[0] * k2[0];
                    sum0 += r2[1] * k2[1];
                    sum0 += r2[2] * k2[2];

                    *destptr0 += sum0;

                    r0++;
                    r1++;
                    r2++;
                    destptr0++;
                }
                r0 +=2;
                r1 += 2;
                r2 += 2;
            }
            kernel0 += 9;
        }
    }         




}