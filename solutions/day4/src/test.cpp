#include <stdio.h>
#include <string.h>
int main()
{
    char a[20]="Program";
    size_t len = strlen(a);
    float x = 2.0f;
    int val = len * x;
    printf("%f\n", val);
}