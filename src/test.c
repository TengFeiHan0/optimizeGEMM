#include <stdio.h>


typedef struct{
    int a[2];
    double d;
}strcut_t;

double fun(int i){

    volatile strcut_t s;
    s.d = 3.14;
    s.a[i] = 1073741824;
    return s.d;
}

int main(){
    double res = fun(6);
    printf("%f", res);
}