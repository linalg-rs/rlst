#include "f2c.h"
#include <stdlib.h>

int xerbla_array_(char *srname, integer *info, int len) {
    char *n = (char*)malloc(sizeof(char)*(len+1));
    memcpy(n, srname, len*sizeof(char));
    n[len]=0;
    printf("** On entry to %6s, parameter number %2i had an illegal value\n",
		n, *info);
    free(n);
    return 0;
}
