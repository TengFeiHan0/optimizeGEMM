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
# TVM
## How to optimize GEMM
All experiments were done under same conditions(target = 'llvm', bn=32, etc)
| Method    | Numpy Time | TVM Time | 
| :------: |:------: |:------: |
| baseline | 0.008193 | 1.987586 | 
| blocking | 0.008379 | 0.218961 | 
| vectorize | 0.008022 | 0.237825 | 
| loop permute| 0.008263 | 0.103705 |
| packing | 0.008152 | 0.104551 | 
| write cache |  0.008472 |  0.099767 | 
| parallel | 0.008281  | 0.032557  | 
| auto-tvm | 0.007933 |  0.007097 | 
