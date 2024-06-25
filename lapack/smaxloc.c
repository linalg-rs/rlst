#define __LAPACK_PRECISION_SINGLE
#include "f2c.h"
integer smaxloc_(scalar *w, integer s, integer e, integer *n)
{
	scalar m; integer i, mi;
	for(m=w[s-1], mi=s, i=s+1; i<=e; i++)
		if (w[i-1]>m) mi=i ,m=w[i-1];
	return mi-s+1;
}
