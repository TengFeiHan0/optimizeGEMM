#include <stdio.h>
#include <mmintrin.h>
#include <xmmintrin.h>  // SSE
#include <pmmintrin.h>  // SSE2
#include <emmintrin.h>  // SSE3
#include <immintrin.h>   //AVX

void print_matrix( int m, int n, double *a, int lda ){
  int i, j;

  for ( j=0; j<n; j++ ){
    for ( i=0; i<m; i++ )
      printf("%le ", a[j*lda +i] );
    printf("\n");
  }
  printf("\n");
}

void matrix_Multiply_1(int m, int n, int k, 
                        double *a , int lda,
                        double *b, int ldb,
                        double *c, int ldc){
        int i, j, p;
        for ( i=0; i<m; i++ ){        /* Loop over the rows of C */
            for ( j=0; j<n; j++ ){        /* Loop over the columns of C */
                for ( p=0; p<k; p++ ){        
                    c[j*ldc + i] = a[j * lda + i] * b[j* ldb + i];
                }
            }
        }                   
}


void matrix_Multiply_2( int m, int n, int k,
                        double *a , int lda,
                        double *b, int ldb,
                        double *c, int ldc){
        int i, j;
        for(j=0; j<n; j++){
            for(i=0; i<m; i++){
                addDot(k, &a[i], lda, &b[j*ldb], ldb, c[j*ldc + i]);
            }
        }
}

void addDot(int k, double *x, int incx, double *y, double *gamma){
    int p;
    for(p=0; p<k; p++){
        *gamma += x[p * incx]*y[p];
    }
}


void matrix_Multiply_3( int m, int n, int k,
                        double *a, int lda,
                        double *b, int ldb,
                        double *c, int ldc){
        int i, j;
        for(j=0; j<n; j +=4){
            for(i=0; i<m; i+=1){
                addDot(k, &a[i], lda, &b[j* ldb], &c[j*ldc + i]);
                addDot(k, &a[i], lda, &b[(j+1)* ldb], &c[(j+1)*ldc + i]);
                addDot(k, &a[i], lda, &b[(j+2)* ldb], &c[(j+2)*ldc + i]);
                addDot(k, &a[i], lda, &b[(j+3)* ldb], &c[(j+3)*ldc + i]);
            }
        }
}

void addDot1x4(int k, double *a, int lda, double *b, int ldb, double *c, int ldc){
    // addDot(k, &a[0], lda, &b[0], &c[0]);
    // addDot(k, &a[0], lda, &b[2*ldb], &c[2*ldc]);
    // addDot(k, &a[0], lda, &b[4*ldb], &c[4*ldc]);
    // addDot(k, &a[0], lda, &b[6*ldb], &c[6*ldc]);

    int p;
    for (p = 0; p <k; p++ ){
        c[0] +=a[p*lda] * b[p];
        c[ldc] += a[p*lda] * b[p+ ldb];
        c[ldc*2] += a[p*lda] * b[p + ldb*2];
        c[ldc*3] += a[p*lda] * b[p + ldb*3];
    }
}


void matrix_Multiply_4(int m, int n, int k,
                        double *a, int lda,
                        double *b, int ldb,
                        double *c, int ldc){
        int i, j;

        for ( j=0; j<n; j+=4 ){        /* Loop over the columns of C, unrolled by 4 */
            for ( i=0; i<m; i+=1 ){        /* Loop over the rows of C */
            /* Update C( i,j ), C( i,j+1 ), C( i,j+2 ), and C( i,j+3 ) in
            one routine (four inner products) */
                addDot1x4( k, &a[i], lda, &b[ldb*j], ldb, &c[ldc*j+i], ldc );
            }
        }                    
                          
                                             
}


void addDot4x4_withReg(int k, double *a, int lda, double *b, int ldb, double *c, int ldc){
    int p;
    register double c_00_reg, c_01_reg, c_02_reg, c_03_reg, a_0p_reg;

    double *bp0_pntr, *bp1_pntr, *bp2_pntr, *bp3_pntr;
    
    c_00_reg = 0.0; 
    c_01_reg = 0.0; 
    c_02_reg = 0.0; 
    c_03_reg = 0.0;

    bp0_pntr = &b[0];
    bp1_pntr = &b[ldb];
    bp2_pntr = &b[ldb*2];
    bp3_pntr = &b[ldb*3];
     for ( p=0; p<k; p+=4 ){
    a_0p_reg = a[lda*p];

    c_00_reg += a_0p_reg * *bp0_pntr;
    c_01_reg += a_0p_reg * *bp1_pntr;
    c_02_reg += a_0p_reg * *bp2_pntr;
    c_03_reg += a_0p_reg * *bp3_pntr;

    a_0p_reg = a[lda*(p+1)];

    c_00_reg += a_0p_reg * *(bp0_pntr+1);
    c_01_reg += a_0p_reg * *(bp1_pntr+1);
    c_02_reg += a_0p_reg * *(bp2_pntr+1);
    c_03_reg += a_0p_reg * *(bp3_pntr+1);

    a_0p_reg = a[lda*(p+2)];

    c_00_reg += a_0p_reg * *(bp0_pntr+2);
    c_01_reg += a_0p_reg * *(bp1_pntr+2);
    c_02_reg += a_0p_reg * *(bp2_pntr+2);
    c_03_reg += a_0p_reg * *(bp3_pntr+2);

    a_0p_reg = a[lda*(p*3)];

    c_00_reg += a_0p_reg * *(bp0_pntr+3);
    c_01_reg += a_0p_reg * *(bp1_pntr+3);
    c_02_reg += a_0p_reg * *(bp2_pntr+3);
    c_03_reg += a_0p_reg * *(bp3_pntr+3);

    bp0_pntr+=4;
    bp1_pntr+=4;
    bp2_pntr+=4;
    bp3_pntr+=4;
  }

  c[0]     += c_00_reg; 
  c[ldc]   += c_01_reg; 
  c[ldc*2] += c_02_reg; 
  c[ldc*3] += c_03_reg;
}


void matrix_Multiply_5( int m, int n, int k, double *a, int lda, 
                                    double *b, int ldb,
                                    double *c, int ldc )
{
  int i, j;

  for ( j=0; j<n; j+=4 ){        /* Loop over the columns of C, unrolled by 4 */
    for ( i=0; i<m; i+=4 ){        /* Loop over the rows of C */
      /* Update C( i,j ), C( i,j+1 ), C( i,j+2 ), and C( i,j+3 ) in
	 one routine (four inner products) */
      addDot4x4_withReg( k, &a[i], lda, &b[j*ldb], ldb, &c[j*ldc +i], ldc );
    }
  }
}

typedef union
{
  __m128d v;
  double d[2];
} v128_d2;

void addDot4x4_withSSE(int k, double *a, int lda, double *b, int ldb, double *c, int ldc){
    int p;
    v128_d2 c_00_c_10_vreg,    c_01_c_11_vreg,    c_02_c_12_vreg,    c_03_c_13_vreg,
    c_20_c_30_vreg,    c_21_c_31_vreg,    c_22_c_32_vreg,    c_23_c_33_vreg,
    a_0p_a_1p_vreg,
    a_2p_a_3p_vreg,
    b_p0_vreg, b_p1_vreg, b_p2_vreg, b_p3_vreg;

    double   *b_p0_pntr, *b_p1_pntr, *b_p2_pntr, *b_p3_pntr;
    b_p0_pntr = &b[0];
    b_p1_pntr = &b[ldb];
    b_p2_pntr = &b[ldb*2];
    b_p3_pntr = &b[ldb*3];

    c_00_c_10_vreg.v = _mm_setzero_pd();   
    c_01_c_11_vreg.v = _mm_setzero_pd();
    c_02_c_12_vreg.v = _mm_setzero_pd(); 
    c_03_c_13_vreg.v = _mm_setzero_pd(); 
    c_20_c_30_vreg.v = _mm_setzero_pd();   
    c_21_c_31_vreg.v = _mm_setzero_pd();  
    c_22_c_32_vreg.v = _mm_setzero_pd();   
    c_23_c_33_vreg.v = _mm_setzero_pd();

    for ( p=0; p<k; p++ ){
    a_0p_a_1p_vreg.v = _mm_load_pd( (double *) &a[lda*p]);
    a_2p_a_3p_vreg.v = _mm_load_pd( (double *) &a[lda*p+2]);

    b_p0_vreg.v = _mm_loaddup_pd( (double *) b_p0_pntr++ );   /* load and duplicate */
    b_p1_vreg.v = _mm_loaddup_pd( (double *) b_p1_pntr++ );   /* load and duplicate */
    b_p2_vreg.v = _mm_loaddup_pd( (double *) b_p2_pntr++ );   /* load and duplicate */
    b_p3_vreg.v = _mm_loaddup_pd( (double *) b_p3_pntr++ );   /* load and duplicate */

    /* First row and second rows */
    c_00_c_10_vreg.v += a_0p_a_1p_vreg.v * b_p0_vreg.v;
    c_01_c_11_vreg.v += a_0p_a_1p_vreg.v * b_p1_vreg.v;
    c_02_c_12_vreg.v += a_0p_a_1p_vreg.v * b_p2_vreg.v;
    c_03_c_13_vreg.v += a_0p_a_1p_vreg.v * b_p3_vreg.v;

    /* Third and fourth rows */
    c_20_c_30_vreg.v += a_2p_a_3p_vreg.v * b_p0_vreg.v;
    c_21_c_31_vreg.v += a_2p_a_3p_vreg.v * b_p1_vreg.v;
    c_22_c_32_vreg.v += a_2p_a_3p_vreg.v * b_p2_vreg.v;
    c_23_c_33_vreg.v += a_2p_a_3p_vreg.v * b_p3_vreg.v;
  }

  c[0] += c_00_c_10_vreg.d[0];  c[ldc] += c_01_c_11_vreg.d[0];  
  c[ldc*2] += c_02_c_12_vreg.d[0];  c[ldc*3] += c_03_c_13_vreg.d[0]; 

  c[1] += c_00_c_10_vreg.d[1];  c[ldc +1] += c_01_c_11_vreg.d[1];  
  c[ldc*2+1] += c_02_c_12_vreg.d[1];  c[ldc*3+1] += c_03_c_13_vreg.d[1]; 

  c[2] += c_20_c_30_vreg.d[0];  c[ldc+2] += c_21_c_31_vreg.d[0];  
  c[ldc*2+2] += c_22_c_32_vreg.d[0];  c[ldc*3+2] += c_23_c_33_vreg.d[0]; 

  c[3] += c_20_c_30_vreg.d[1];  c[ldc +3] += c_21_c_31_vreg.d[1];  
  c[ldc*2 +3] += c_22_c_32_vreg.d[1];  c[ldc*3 +3] += c_23_c_33_vreg.d[1];   

}

typedef union
{
  __m256d v;
  double d[4];
} v256_d4;


void matrix_Multiply_6( int m, int n, int k, double *a, int lda, 
                                    double *b, int ldb,
                                    double *c, int ldc )
{
  int i, j;

  for ( j=0; j<n; j+=4 ){        /* Loop over the columns of C, unrolled by 4 */
    for ( i=0; i<m; i+=4 ){        /* Loop over the rows of C */
      /* Update C( i,j ), C( i,j+1 ), C( i,j+2 ), and C( i,j+3 ) in
	 one routine (four inner products) */
      addDot4x4_withSSE( k, &a[i], lda, &b[j*ldb], ldb, &c[j*ldc +i], ldc );
    }
  }
}


void addDot4x4_withAVX( int k, double *a, int lda, double *b, int ldb, double *c, int ldc){
    
    int p;

    v256_d4 c_00_c_10_c_20_c_30_vreg;// column 0 of c
    v256_d4 c_01_c_11_c_21_c_31_vreg;// column 1 of c
    v256_d4 c_02_c_12_c_22_c_32_vreg;// column 2 of c
    v256_d4 c_03_c_13_c_23_c_33_vreg;// colunm 3 of c
    v256_d4 b_p0_vreg, b_p1_vreg, b_p2_vreg, b_p3_vreg; 

    v256_d4 a_0p_a_1p_a_2p_a_3p;// one colunm of A
    double *b_p0_pntr, *b_p1_pntr, *b_p2_pntr, *b_p3_pntr; 
    
    b_p0_pntr = &b[0];
    b_p1_pntr = &b[ldb];
    b_p2_pntr = &b[ldb*2];
    b_p3_pntr = &b[ldb*3];

    c_00_c_10_c_20_c_30_vreg.v = _mm256_setzero_pd();   
    c_01_c_11_c_21_c_31_vreg.v = _mm256_setzero_pd();  
    c_02_c_12_c_22_c_32_vreg.v = _mm256_setzero_pd();  
    c_03_c_13_c_23_c_33_vreg.v = _mm256_setzero_pd();

    for ( p=0; p<k; p++ ){
    a_0p_a_1p_a_2p_a_3p.v = _mm256_loadu_pd( (double *) &a[p*lda]);

    b_p0_vreg.v = _mm256_broadcast_sd( (double *) b_p0_pntr++ ); 
    b_p1_vreg.v = _mm256_broadcast_sd( (double *) b_p1_pntr++ ); 
    b_p2_vreg.v = _mm256_broadcast_sd( (double *) b_p2_pntr++ ); 
    b_p3_vreg.v = _mm256_broadcast_sd( (double *) b_p3_pntr++ ); 
  
    c_00_c_10_c_20_c_30_vreg.v = _mm256_fmadd_pd (b_p0_vreg.v, a_0p_a_1p_a_2p_a_3p.v, c_00_c_10_c_20_c_30_vreg.v);
    c_01_c_11_c_21_c_31_vreg.v = _mm256_fmadd_pd (b_p1_vreg.v, a_0p_a_1p_a_2p_a_3p.v, c_01_c_11_c_21_c_31_vreg.v);
    c_02_c_12_c_22_c_32_vreg.v = _mm256_fmadd_pd (b_p2_vreg.v, a_0p_a_1p_a_2p_a_3p.v, c_02_c_12_c_22_c_32_vreg.v);
    c_03_c_13_c_23_c_33_vreg.v = _mm256_fmadd_pd (b_p3_vreg.v, a_0p_a_1p_a_2p_a_3p.v, c_03_c_13_c_23_c_33_vreg.v);
  }

  c[0] += c_00_c_10_c_20_c_30_vreg.d[0];
  c[1] += c_00_c_10_c_20_c_30_vreg.d[1];
  c[2] += c_00_c_10_c_20_c_30_vreg.d[2];
  c[3] += c_00_c_10_c_20_c_30_vreg.d[3];

  c[ldc] += c_01_c_11_c_21_c_31_vreg.d[0];
  c[ldc +1] += c_01_c_11_c_21_c_31_vreg.d[1];
  c[ldc +2] += c_01_c_11_c_21_c_31_vreg.d[2];
  c[ldc +3] += c_01_c_11_c_21_c_31_vreg.d[3];

  c[2*ldc] += c_02_c_12_c_22_c_32_vreg.d[0];
  c[2*ldc+1] += c_02_c_12_c_22_c_32_vreg.d[1];
  c[2*ldc+2] += c_02_c_12_c_22_c_32_vreg.d[2];
  c[2*ldc+3] += c_02_c_12_c_22_c_32_vreg.d[3];

  c[3*ldc] += c_03_c_13_c_23_c_33_vreg.d[0];
  c[3*ldc+1] += c_03_c_13_c_23_c_33_vreg.d[1];
  c[3*ldc+2] += c_03_c_13_c_23_c_33_vreg.d[2];
  c[3*ldc+3] += c_03_c_13_c_23_c_33_vreg.d[3];
}

void matrix_Multiply_7( int m, int n, int k, double *a, int lda, 
                                    double *b, int ldb,
                                    double *c, int ldc )
{
  int i, j;

  for ( j=0; j<n; j+=4 ){        /* Loop over the columns of C, unrolled by 4 */
    for ( i=0; i<m; i+=4 ){        /* Loop over the rows of C */
      /* Update C( i,j ), C( i,j+1 ), C( i,j+2 ), and C( i,j+3 ) in
	 one routine (four inner products) */
      addDot4x4_withAVX( k, &a[i], lda, &b[j*ldb], ldb, &c[j*ldc +i], ldc );
    }
  }
}

#define mc 256
#define kc 128


void matrix_Multiply_8( int m, int n, int k, double *a, int lda, 
                                    double *b, int ldb,
                                    double *c, int ldc )
{
  int i, j, p, pb, ib;

  /* This time, we compute a mc x n block of C by a call to the InnerKernel */

  for ( p=0; p<k; p+=kc ){
    pb = min( k-p, kc );
    for ( i=0; i<m; i+=mc ){
      ib = min( m-i, mc );
      InnerKernel( ib, n, pb, &a[i +p*lda], lda, &b[p], ldb, &c[i], ldc );
    }
  }
}

void InnerKernel( int m, int n, int k, double *a, int lda, 
                                       double *b, int ldb,
                                       double *c, int ldc )
{
  int i, j;

  for ( j=0; j<n; j+=4 ){        /* Loop over the columns of C, unrolled by 4 */
    for ( i=0; i<m; i+=4 ){        /* Loop over the rows of C */
      /* Update C( i,j ), C( i,j+1 ), C( i,j+2 ), and C( i,j+3 ) in
	 one routine (four inner products) */

      addDot4x4_withAVX( k, &a[i], lda, &b[j*ldb], ldb, &c[i+ j*ldc], ldc );
    }
  }
}




