#include "f2c.h"

void xerbla_(char *srname, integer *info) {
    printf("** On entry to %6s, parameter number %2i had an illegal value\n",
		srname, *info);
}
