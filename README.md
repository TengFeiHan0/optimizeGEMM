# optimizeGEMM

## conv3x3s1
|  number of thread    | time | 
| :------: |:------:  |
|  0 | 3773.15 |     
|  2 | 1912.93|   
| 4 | 1050.26 |
| 8  | 758.06 |

## Im2col+sgemm
|  number of thread    | time | 
| :------: |:------:  |
|  0 | 2649.02 |
| 0+AVX | 213.42 |
|  2 | 1395.93|   
| 4 | 770.47 |
| 8  | 695.93 |
| 8+AVX | 48.42 |

# how to optimize GEMM(TVM)

| Method    | Numpy Time | TVM Time | FLOPs |
| :------: |:------: |:------: |:------: |
| |  | | |
| |  | | |
| | |  | | 
| |  | | |
| |  | | |
|  |  | | |
