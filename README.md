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
|  with AVX | 213.42 |
|  2 | 1395.93|   
| 4 | 770.47 |
| 8  | 695.93 |
|  with AVX | 48.42 |
