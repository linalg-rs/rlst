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

/* > \brief \b CLAPLL measures the linear dependence of two vectors. */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download CLAPLL + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/clapll.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/clapll.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/clapll.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE CLAPLL( N, X, INCX, Y, INCY, SSMIN ) */

/*       INTEGER            INCX, INCY, N */
/*       REAL               SSMIN */
/*       COMPLEX            X( * ), Y( * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > Given two column vectors X and Y, let */
/* > */
/* >                      A = ( X Y ). */
/* > */
/* > The subroutine first computes the QR factorization of A = Q*R, */
/* > and then computes the SVD of the 2-by-2 upper triangular matrix R. */
/* > The smaller singular value of R is returned in SSMIN, which is used */
/* > as the measurement of the linear dependency of the vectors X and Y. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >          The length of the vectors X and Y. */
/* > \endverbatim */
/* > */
/* > \param[in,out] X */
/* > \verbatim */
/* >          X is COMPLEX array, dimension (1+(N-1)*INCX) */
/* >          On entry, X contains the N-vector X. */
/* >          On exit, X is overwritten. */
/* > \endverbatim */
/* > */
/* > \param[in] INCX */
/* > \verbatim */
/* >          INCX is INTEGER */
/* >          The increment between successive elements of X. INCX > 0. */
/* > \endverbatim */
/* > */
/* > \param[in,out] Y */
/* > \verbatim */
/* >          Y is COMPLEX array, dimension (1+(N-1)*INCY) */
/* >          On entry, Y contains the N-vector Y. */
/* >          On exit, Y is overwritten. */
/* > \endverbatim */
/* > */
/* > \param[in] INCY */
/* > \verbatim */
/* >          INCY is INTEGER */
/* >          The increment between successive elements of Y. INCY > 0. */
/* > \endverbatim */
/* > */
/* > \param[out] SSMIN */
/* > \verbatim */
/* >          SSMIN is REAL */
/* >          The smallest singular value of the N-by-2 matrix A = ( X Y ). */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date December 2016 */

/* > \ingroup complexOTHERauxiliary */

/*  ===================================================================== */
void  clapll_(integer *n, complex *x, integer *incx, complex *
	y, integer *incy, real *ssmin)
{
    /* System generated locals */
    integer i__1;
    real r__1, r__2, r__3;
    complex q__1, q__2, q__3, q__4;

    /* Local variables */
    complex c__, a11, a12, a22, tau;
    extern void  slas2_(real *, real *, real *, real *, real *)
	    ;
    extern /* Complex */ VOID cdotc_(complex *, integer *, complex *, integer 
	    *, complex *, integer *);
    extern void  caxpy_(integer *, complex *, complex *, 
	    integer *, complex *, integer *);
    real ssmax;
    extern void  clarfg_(integer *, complex *, complex *, 
	    integer *, complex *);


/*  -- LAPACK auxiliary routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     December 2016 */


/*  ===================================================================== */


/*     Quick return if possible */

    /* Parameter adjustments */
    --y;
    --x;

    /* Function Body */
    if (*n <= 1) {
	*ssmin = 0.f;
	return;
    }

/*     Compute the QR factorization of the N-by-2 matrix ( X Y ) */

    clarfg_(n, &x[1], &x[*incx + 1], incx, &tau);
    a11.r = x[1].r, a11.i = x[1].i;
    x[1].r = 1.f, x[1].i = 0.f;

    r_cnjg(&q__3, &tau);
    q__2.r = -q__3.r, q__2.i = -q__3.i;
    cdotc_(&q__4, n, &x[1], incx, &y[1], incy);
    q__1.r = q__2.r * q__4.r - q__2.i * q__4.i, q__1.i = q__2.r * q__4.i + 
	    q__2.i * q__4.r;
    c__.r = q__1.r, c__.i = q__1.i;
    caxpy_(n, &c__, &x[1], incx, &y[1], incy);

    i__1 = *n - 1;
    clarfg_(&i__1, &y[*incy + 1], &y[(*incy << 1) + 1], incy, &tau);

    a12.r = y[1].r, a12.i = y[1].i;
    i__1 = *incy + 1;
    a22.r = y[i__1].r, a22.i = y[i__1].i;

/*     Compute the SVD of 2-by-2 Upper triangular matrix. */

    r__1 = c_abs(&a11);
    r__2 = c_abs(&a12);
    r__3 = c_abs(&a22);
    slas2_(&r__1, &r__2, &r__3, ssmin, &ssmax);

    return;

/*     End of CLAPLL */

} /* clapll_ */

