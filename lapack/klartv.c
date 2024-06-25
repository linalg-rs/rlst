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

#define __LAPACK_PRECISION_HALF
#include "f2c.h"

/* > \brief \b ZLARTV applies a vector of plane rotations with doublereal cosines and complex sines to the elements 
of a pair of vectors. */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download ZLARTV + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/zlartv.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/zlartv.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/zlartv.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE ZLARTV( N, X, INCX, Y, INCY, C, S, INCC ) */

/*       INTEGER            INCC, INCX, INCY, N */
/*       DOUBLE PRECISION   C( * ) */
/*       COMPLEX*16         S( * ), X( * ), Y( * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > ZLARTV applies a vector of complex plane rotations with doublereal cosines */
/* > to elements of the complex vectors x and y. For i = 1,2,...,n */
/* > */
/* >    ( x(i) ) := (        c(i)   s(i) ) ( x(i) ) */
/* >    ( y(i) )    ( -conjg(s(i))  c(i) ) ( y(i) ) */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >          The number of plane rotations to be applied. */
/* > \endverbatim */
/* > */
/* > \param[in,out] X */
/* > \verbatim */
/* >          X is COMPLEX*16 array, dimension (1+(N-1)*INCX) */
/* >          The vector x. */
/* > \endverbatim */
/* > */
/* > \param[in] INCX */
/* > \verbatim */
/* >          INCX is INTEGER */
/* >          The increment between elements of X. INCX > 0. */
/* > \endverbatim */
/* > */
/* > \param[in,out] Y */
/* > \verbatim */
/* >          Y is COMPLEX*16 array, dimension (1+(N-1)*INCY) */
/* >          The vector y. */
/* > \endverbatim */
/* > */
/* > \param[in] INCY */
/* > \verbatim */
/* >          INCY is INTEGER */
/* >          The increment between elements of Y. INCY > 0. */
/* > \endverbatim */
/* > */
/* > \param[in] C */
/* > \verbatim */
/* >          C is DOUBLE PRECISION array, dimension (1+(N-1)*INCC) */
/* >          The cosines of the plane rotations. */
/* > \endverbatim */
/* > */
/* > \param[in] S */
/* > \verbatim */
/* >          S is COMPLEX*16 array, dimension (1+(N-1)*INCC) */
/* >          The sines of the plane rotations. */
/* > \endverbatim */
/* > */
/* > \param[in] INCC */
/* > \verbatim */
/* >          INCC is INTEGER */
/* >          The increment between elements of C and S. INCC > 0. */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date December 2016 */

/* > \ingroup complex16OTHERauxiliary */

/*  ===================================================================== */
void  klartv_(integer *n, halfcomplex *x, integer *incx, 
	halfcomplex *y, integer *incy, halfreal *c__, halfcomplex *s, 
	integer *incc)
{
    /* System generated locals */
    integer i__1, i__2, i__3, i__4;
    halfcomplex z__1, z__2, z__3, z__4;

    /* Local variables */
    integer i__, ic, ix, iy;
    halfcomplex xi, yi;


/*  -- LAPACK auxiliary routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     December 2016 */


/*  ===================================================================== */


    /* Parameter adjustments */
    --s;
    --c__;
    --y;
    --x;

    /* Function Body */
    ix = 1;
    iy = 1;
    ic = 1;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	i__2 = ix;
	xi.r = x[i__2].r, xi.i = x[i__2].i;
	i__2 = iy;
	yi.r = y[i__2].r, yi.i = y[i__2].i;
	i__2 = ix;
	i__3 = ic;
	z__2.r = c__[i__3] * xi.r, z__2.i = c__[i__3] * xi.i;
	i__4 = ic;
	z__3.r = s[i__4].r * yi.r - s[i__4].i * yi.i, z__3.i = s[i__4].r * 
		yi.i + s[i__4].i * yi.r;
	z__1.r = z__2.r + z__3.r, z__1.i = z__2.i + z__3.i;
	x[i__2].r = z__1.r, x[i__2].i = z__1.i;
	i__2 = iy;
	i__3 = ic;
	z__2.r = c__[i__3] * yi.r, z__2.i = c__[i__3] * yi.i;
	d_cnjg(&z__4, &s[ic]);
	z__3.r = z__4.r * xi.r - z__4.i * xi.i, z__3.i = z__4.r * xi.i + 
		z__4.i * xi.r;
	z__1.r = z__2.r - z__3.r, z__1.i = z__2.i - z__3.i;
	y[i__2].r = z__1.r, y[i__2].i = z__1.i;
	ix += *incx;
	iy += *incy;
	ic += *incc;
/* L10: */
    }
    return;

/*     End of ZLARTV */

} /* klartv_ */

