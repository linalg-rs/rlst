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

#define __LAPACK_PRECISION_SINGLE
#include "f2c.h"

/* > \brief \b SCSUM1 forms the 1-norm of the complex vector using the true absolute value. */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download SCSUM1 + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/scsum1.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/scsum1.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/scsum1.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       REAL             FUNCTION SCSUM1( N, CX, INCX ) */

/*       INTEGER            INCX, N */
/*       COMPLEX            CX( * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > SCSUM1 takes the sum of the absolute values of a complex */
/* > vector and returns a single precision result. */
/* > */
/* > Based on SCASUM from the Level 1 BLAS. */
/* > The change is to use the 'genuine' absolute value. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >          The number of elements in the vector CX. */
/* > \endverbatim */
/* > */
/* > \param[in] CX */
/* > \verbatim */
/* >          CX is COMPLEX array, dimension (N) */
/* >          The vector whose elements will be summed. */
/* > \endverbatim */
/* > */
/* > \param[in] INCX */
/* > \verbatim */
/* >          INCX is INTEGER */
/* >          The spacing between successive values of CX.  INCX > 0. */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date December 2016 */

/* > \ingroup complexOTHERauxiliary */

/* > \par Contributors: */
/*  ================== */
/* > */
/* > Nick Higham for use with CLACON. */

/*  ===================================================================== */
real scsum1_(integer *n, complex *cx, integer *incx)
{
    /* System generated locals */
    integer i__1, i__2;
    real ret_val;

    /* Local variables */
    integer i__, nincx;
    real stemp;


/*  -- LAPACK auxiliary routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     December 2016 */


/*  ===================================================================== */


    /* Parameter adjustments */
    --cx;

    /* Function Body */
    ret_val = 0.f;
    stemp = 0.f;
    if (*n <= 0) {
	return ret_val;
    }
    if (*incx == 1) {
	goto L20;
    }

/*     CODE FOR INCREMENT NOT EQUAL TO 1 */

    nincx = *n * *incx;
    i__1 = nincx;
    i__2 = *incx;
    for (i__ = 1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ += i__2) {

/*        NEXT LINE MODIFIED. */

	stemp += c_abs(&cx[i__]);
/* L10: */
    }
    ret_val = stemp;
    return ret_val;

/*     CODE FOR INCREMENT EQUAL TO 1 */

L20:
    i__2 = *n;
    for (i__ = 1; i__ <= i__2; ++i__) {

/*        NEXT LINE MODIFIED. */

	stemp += c_abs(&cx[i__]);
/* L30: */
    }
    ret_val = stemp;
    return ret_val;

/*     End of SCSUM1 */

} /* scsum1_ */

