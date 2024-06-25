#define len_trim__(cad,len) ({                           integer _r=0,i;                           for(i=0; i<(len) && (cad)[i]; i++)                             if((cad)[i] != ' ') _r=i;                           _r+1; })
#define ceiling_(a) (myceil(*(a)))
#define myceil(a) (sizeof(a) == sizeof(float) ? ceilf(a) : ceil(a))
#include <math.h>
/*  -- translated by f2c (version 20200916).
   You must link the resulting object file with libf2c:
	on Microsoft Windows system, link with libf2c.lib;
	on Linux or Unix systems, link with .../path/to/libf2c.a -lm
	or, if you install libf2c.a in a standard place, with -lf2c -lm
	-- in that order, at the end of the command line, as in
		cc *.o -lf2c -lm
	Source for libf2c is in /netlib/f2c/libf2c.zip, e.g.,

		http://www.netlib.org/f2c/libf2c.zip
*/

#define __LAPACK_PRECISION_DOUBLE
#include "f2c.h"

/* > \brief \b IZMAX1 finds the index of the first vector element of maximum absolute value. */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download IZMAX1 + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/izmax1.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/izmax1.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/izmax1.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       INTEGER          FUNCTION IZMAX1( N, ZX, INCX ) */

/*       INTEGER            INCX, N */
/*       COMPLEX*16         ZX( * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > IZMAX1 finds the index of the first vector element of maximum absolute value. */
/* > */
/* > Based on IZAMAX from Level 1 BLAS. */
/* > The change is to use the 'genuine' absolute value. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >          The number of elements in the vector ZX. */
/* > \endverbatim */
/* > */
/* > \param[in] ZX */
/* > \verbatim */
/* >          ZX is COMPLEX*16 array, dimension (N) */
/* >          The vector ZX. The IZMAX1 function returns the index of its first */
/* >          element of maximum absolute value. */
/* > \endverbatim */
/* > */
/* > \param[in] INCX */
/* > \verbatim */
/* >          INCX is INTEGER */
/* >          The spacing between successive values of ZX.  INCX >= 1. */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date February 2014 */

/* > \ingroup complexOTHERauxiliary */

/* > \par Contributors: */
/*  ================== */
/* > */
/* > Nick Higham for use with ZLACON. */

/*  ===================================================================== */
integer izmax1_(integer *n, doublecomplex *zx, integer *incx)
{
    /* System generated locals */
    integer ret_val, i__1;

    /* Local variables */
    integer i__, ix;
    doublereal dmax__;


/*  -- LAPACK auxiliary routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     February 2014 */


/*  ===================================================================== */


    /* Parameter adjustments */
    --zx;

    /* Function Body */
    ret_val = 0;
    if (*n < 1 || *incx <= 0) {
	return ret_val;
    }
    ret_val = 1;
    if (*n == 1) {
	return ret_val;
    }
    if (*incx == 1) {

/*        code for increment equal to 1 */

	dmax__ = z_abs(&zx[1]);
	i__1 = *n;
	for (i__ = 2; i__ <= i__1; ++i__) {
	    if (z_abs(&zx[i__]) > dmax__) {
		ret_val = i__;
		dmax__ = z_abs(&zx[i__]);
	    }
	}
    } else {

/*        code for increment not equal to 1 */

	ix = 1;
	dmax__ = z_abs(&zx[1]);
	ix += *incx;
	i__1 = *n;
	for (i__ = 2; i__ <= i__1; ++i__) {
	    if (z_abs(&zx[ix]) > dmax__) {
		ret_val = i__;
		dmax__ = z_abs(&zx[ix]);
	    }
	    ix += *incx;
	}
    }
    return ret_val;

/*     End of IZMAX1 */

} /* izmax1_ */

