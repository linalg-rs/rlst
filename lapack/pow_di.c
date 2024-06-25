#define __LAPACK_PRECISION_DOUBLE
#include "f2c.h"
dscalar dpow_ui(scalar *_x, integer n) {
	dscalar x = *_x; dscalar pow=1.0; unsigned long int u;
	if(n != 0) {
		if(n < 0) n = -n, x = 1/x;
		for(u = n; ; ) {
			if(u & 01) pow *= x;
			if(u >>= 1) x *= x;
			else break;
		}
	}
	return pow;
}
