#define __LAPACK_PRECISION_SINGLE
#include "f2c.h"
scalar sf__cabs(scalar r, scalar i) {
	scalar temp;
	if(r < 0) r = -r;
	if(i < 0) i = -i;
	if(i > r){
		temp = r;
		r = i;
		i = temp;
	}
	if((r+i) == r)
		temp = r;
	else {
		temp = i/r;
		temp = r*M(sqrt)(M(1.0) + temp*temp);  /*overflow!!*/
	}
	return temp;
}
