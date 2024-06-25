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

#define __LAPACK_PRECISION_QUAD
#include "f2c.h"

/* > \brief \b DLARNV returns a vector of random numbers from a uniform or normal distribution. */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download DLARNV + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlarnv.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlarnv.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlarnv.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE DLARNV( IDIST, ISEED, N, X ) */

/*       INTEGER            IDIST, N */
/*       INTEGER            ISEED( 4 ) */
/*       DOUBLE PRECISION   X( * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > DLARNV returns a vector of n random doublereal numbers from a uniform or */
/* > normal distribution. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] IDIST */
/* > \verbatim */
/* >          IDIST is INTEGER */
/* >          Specifies the distribution of the random numbers: */
/* >          = 1:  uniform (0,1) */
/* >          = 2:  uniform (-1,1) */
/* >          = 3:  normal (0,1) */
/* > \endverbatim */
/* > */
/* > \param[in,out] ISEED */
/* > \verbatim */
/* >          ISEED is INTEGER array, dimension (4) */
/* >          On entry, the seed of the random number generator; the array */
/* >          elements must be between 0 and 4095, and ISEED(4) must be */
/* >          odd. */
/* >          On exit, the seed is updated. */
/* > \endverbatim */
/* > */
/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >          The number of random numbers to be generated. */
/* > \endverbatim */
/* > */
/* > \param[out] X */
/* > \verbatim */
/* >          X is DOUBLE PRECISION array, dimension (N) */
/* >          The generated random numbers. */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date December 2016 */

/* > \ingroup OTHERauxiliary */

/* > \par Further Details: */
/*  ===================== */
/* > */
/* > \verbatim */
/* > */
/* >  This routine calls the auxiliary routine DLARUV to generate random */
/* >  doublereal numbers from a uniform (0,1) distribution, in batches of up to */
/* >  128 using vectorisable code. The Box-Muller method is used to */
/* >  transform numbers from a uniform to a normal distribution. */
/* > \endverbatim */
/* > */
/*  ===================================================================== */
void  qlarnv_(integer *idist, integer *iseed, integer *n, 
	quadreal *x)
{
    /* System generated locals */
    integer i__1, i__2, i__3;

    /* Local variables */
    integer i__;
    quadreal u[128];
    integer il, iv, il2;
    extern void  qlaruv_(integer *, integer *, quadreal *);


/*  -- LAPACK auxiliary routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     December 2016 */


/*  ===================================================================== */


    /* Parameter adjustments */
    --x;
    --iseed;

    /* Function Body */
    i__1 = *n;
    for (iv = 1; iv <= i__1; iv += 64) {
/* Computing MIN */
	i__2 = 64, i__3 = *n - iv + 1;
	il = f2cmin(i__2,i__3);
	if (*idist == 3) {
	    il2 = il << 1;
	} else {
	    il2 = il;
	}

/*        Call DLARUV to generate IL2 numbers from a uniform (0,1) */
/*        distribution (IL2 <= LV) */

	qlaruv_(&iseed[1], &il2, u);

	if (*idist == 1) {

/*           Copy generated numbers */

	    i__2 = il;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		x[iv + i__ - 1] = u[i__ - 1];
/* L10: */
	    }
	} else if (*idist == 2) {

/*           Convert generated numbers to uniform (-1,1) distribution */

	    i__2 = il;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		x[iv + i__ - 1] = u[i__ - 1] * 2. - 1.;
/* L20: */
	    }
	} else if (*idist == 3) {

/*           Convert generated numbers to normal (0,1) distribution */

	    i__2 = il;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		x[iv + i__ - 1] = M(sqrt)(M(log)(u[(i__ << 1) - 2]) * -2.) * M(cos)(u[(
			i__ << 1) - 1] * 6.2831853071795864769252867663);
/* L30: */
	    }
	}
/* L40: */
    }
    return;

/*     End of DLARNV */

} /* qlarnv_ */

