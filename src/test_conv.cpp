#include<iostream>
#include<stdio.h>
#include<math.h>
#include<opencv2/opencv.hpp>
#include <convolution_sgemm.h>
#include <convolution_3x3s1.h>
#include <convolution_3x3s2.h>

using namespace std;
using namespace cv;

float a[200]= {-0.3670,  0.6706, -0.4710, -0.4232,  0.0396,
           1.1935,  1.2700, -0.4183, -0.2690, -1.1960,
          -0.3244,  0.3431, -0.0428, -0.7895, -1.1025,
          -0.3374, -2.0180, -2.8447,  1.1366,  0.9439,
          -1.7559, -0.2478,  0.2211,  1.4843, -0.4977,

          0.7037,  0.1641,  0.5758,  0.4975, -1.5003,
          -0.7084, -0.6419, -0.2333, -2.3270, -0.9529,
          -0.0327,  0.1763, -0.4550, -1.9170,  1.3147,
           0.9061,  1.1292,  0.3392, -0.8160, -0.1567,
           0.6644,  0.1139,  0.1924, -0.6630,  0.7814,

          0.3477, -0.2124, -0.0696, -0.6485,  1.0422,
          -1.5059,  0.6144, -0.9405, -0.3811,  0.4394,
           0.2629, -1.4742, -0.5631,  1.1443, -0.1476,
           1.3373, -1.6746,  0.8771, -2.1727, -1.2477,
          -0.4628, -0.1945, -0.1130, -0.1061,  0.6973};

float b[200] = {-0.17001375555992126, -0.12650497257709503, -0.17024663090705872, -0.1456221491098404, 0.1466996818780899, 
0.03860868513584137, 0.11999060213565826, -0.01779240369796753, 0.12982524931430817, 0.015236392617225647, 
-0.054507315158843994, -0.12870833277702332, 0.02602560818195343, -0.03528741002082825, -0.033567413687705994,
 0.18616314232349396, 0.06957052648067474, 0.04186442494392395, -0.18866780400276184, -0.08342712372541428, 
 -0.01155330240726471, -0.040157049894332886, 0.1788344532251358, 0.07730035483837128, -0.19159328937530518, 0.019566774368286133,
  -0.08382271230220795, -0.10531263053417206, -0.1174154132604599, -0.0275709331035614, 0.11025996506214142, 0.18692992627620697, 
  -0.17520980536937714, 0.18026240170001984, -0.06548972427845001, 0.07031305134296417, 0.0059783607721328735, 0.05703267455101013,
   -0.18433929979801178, 0.025940656661987305, 0.18331675231456757, -0.19137872755527496, -0.10033346712589264, 0.021840929985046387,
    0.15824191272258759, -0.012584224343299866, -0.14148840308189392, 0.1656934767961502, -0.004484876990318298, 0.11100687086582184, 
    0.03759340941905975, 0.033648014068603516, -0.07029666006565094, -0.11860565841197968, -0.03125421702861786, -0.05637654662132263, 
    0.12427116930484772, -0.009501680731773376, 0.023209944367408752, 0.10611219704151154, -0.17002975940704346, 0.16519664227962494, 
    0.1539759486913681, -0.0811036005616188, 0.16470853984355927, -0.10668835788965225, -0.18951310217380524, 0.0592428594827652, 
    -0.06633712351322174, -0.19214779138565063, -0.056283965706825256, 0.1682199388742447, 0.18041475117206573, 0.031057968735694885,
     0.10012434422969818, -0.15559618175029755, -0.04959174990653992, -0.1614183634519577, 0.0902632623910904, 0.11818964779376984,
      0.0028641074895858765, -0.044215574860572815, -0.04329819977283478, -0.13284391164779663, -0.11242090910673141, -0.15806257724761963, 
      -0.03721471130847931, 0.01666019856929779, 0.0674336701631546, -0.15946170687675476, 0.09025247395038605, -0.01745268702507019, 
      -0.07207220047712326, -0.0579836368560791, 0.05527739226818085, 0.1581626683473587, 0.11893303692340851, -0.16182971000671387, 
      0.045212507247924805, -0.17816422879695892, -0.10284984856843948, -0.04865074157714844, -0.07178143411874771, -0.02878934144973755, 
      -0.007202267646789551, 0.15899141132831573, 0.17417331039905548, -0.003684520721435547, 0.16760773956775665, 0.1435248702764511, 
      -0.012662217020988464, -0.18273133039474487, -0.1614561527967453, 0.05265532433986664, 0.1612069457769394, -0.0220462828874588, 
      -0.006873860955238342, 0.01519419252872467, -0.0014906525611877441, -0.16131097078323364, -0.03826208412647247, -0.15336640179157257, 
      -0.0061959028244018555, 0.0039733946323394775, -0.1278531700372696, 0.1167348176240921, -0.09316063672304153, 0.11908359825611115, 
      0.16194234788417816, -0.015045925974845886, 0.08637414872646332, 0.015845999121665955, 0.0645458847284317, 0.16166676580905914, -0.0012359023094177246  };

float c[200]={-0.0511, -0.3954,  0.2891,
          -0.6829,  0.8732,  0.4890,
          -0.4660,  0.8988,  0.2948,

          0.3154, -0.2506,  0.6096,
          -0.5991,  0.6472, -0.4285,
          -0.0155, -0.1286, -1.1570,

          0.1869, -0.6136,  0.4527,
          -1.3240, -0.0717, -0.1254,
          -0.6036,  0.4735, -0.1250,

         -0.4626, -0.5322,  0.7050,
           0.2839, -0.5795,  0.5798,
           0.5173,  0.7638,  0.4037,

         -0.7240, -0.6133,  1.1883,
           0.0380, -0.1077, -0.0369,
          -0.6761,  1.3065,  0.2785};



int main(){
    //input
    const int inw = 112;
    const int inh = 112;
    const int inch = 128;
    //kernel
    const int kw =3;
    const int kh =3;
    //output
    const int outch =128;
    int stride = 1;

    const int outw = (inw - kw)/stride +1;
    const int outh = (inh - kh)/stride +1;
    
    //5x5x3
    float *src = new float[inw * inh * inch];
    //3x3x4
    float *kernel = new float[kw * kh * outch * inch];
    //3x3x4
    float *dest = new float[outw * outh * outch];

    for(int i = 0; i <inw* inh * inch; i++){
        src[i] = 1.0;
    }
    for(int i = 0; i < kw * kh * inch * outch; i++){
        kernel[i] = 1.0;
    }
    int64 st = cv::getTickCount();
    //im2col后的kernel
    float *kernel_temp = new float[kw*kh* inch*(outch/4 +outch%4)*4];
    //do im2col kernel
    convolutionTransformKernel(kernel, kw, kh, kernel_temp, inch, outch);

    
    for(int i = 0; i <10; i++){
        convolutionIm2colSgemm(src, inw, inh, inch, kernel, kernel_temp, 
                       kw, kh, dest, outw, outh, outch, stride, stride);
    }

    // for(int i = 0; i <10; i++){
    //     conv3x3s1(src, inw, inh, inch, kernel, dest, outw, outh, outch);
    // }
    
    double duration = (cv::getTickCount() - st) / cv::getTickFrequency() * 100;
    printf("Time: %.5f\n", duration);

    free(src);
    free(kernel);
    free(dest);
    return 0;
}
