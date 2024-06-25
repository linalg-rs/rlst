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

/* > \brief \b ZPTTRF */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download ZPTTRF + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/zpttrf.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/zpttrf.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/zpttrf.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE ZPTTRF( N, D, E, INFO ) */

/*       INTEGER            INFO, N */
/*       DOUBLE PRECISION   D( * ) */
/*       COMPLEX*16         E( * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > ZPTTRF computes the L*D*L**H factorization of a complex Hermitian */
/* > positive definite tridiagonal matrix A.  The factorization may also */
/* > be regarded as having the form A = U**H *D*U. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >          The order of the matrix A.  N >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in,out] D */
/* > \verbatim */
/* >          D is DOUBLE PRECISION array, dimension (N) */
/* >          On entry, the n diagonal elements of the tridiagonal matrix */
/* >          A.  On exit, the n diagonal elements of the diagonal matrix */
/* >          D from the L*D*L**H factorization of A. */
/* > \endverbatim */
/* > */
/* > \param[in,out] E */
/* > \verbatim */
/* >          E is COMPLEX*16 array, dimension (N-1) */
/* >          On entry, the (n-1) subdiagonal elements of the tridiagonal */
/* >          matrix A.  On exit, the (n-1) subdiagonal elements of the */
/* >          unit bidiagonal factor L from the L*D*L**H factorization of A. */
/* >          E can also be regarded as the superdiagonal of the unit */
/* >          bidiagonal factor U from the U**H *D*U factorization of A. */
/* > \endverbatim */
/* > */
/* > \param[out] INFO */
/* > \verbatim */
/* >          INFO is INTEGER */
/* >          = 0: successful exit */
/* >          < 0: if INFO = -k, the k-th argument had an illegal value */
/* >          > 0: if INFO = k, the leading minor of order k is not */
/* >               positive definite; if k < N, the factorization could not */
/* >               be completed, while if k = N, the factorization was */
/* >               completed, but D(N) <= 0. */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date December 2016 */

/* > \ingroup complex16PTcomputational */

/*  ===================================================================== */
void  wpttrf_(integer *n, quadreal *d__, quadcomplex *e, 
	integer *info)
{
    /* System generated locals */
    integer i__1, i__2;
    quadcomplex z__1;

    /* Local variables */
    quadreal f, g;
    integer i__, i4;
    quadreal eii, eir;
    extern void  xerbla_(char *, integer *);


/*  -- LAPACK computational routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     December 2016 */


/*  ===================================================================== */


/*     Test the input parameters. */

    /* Parameter adjustments */
    --e;
    --d__;

    /* Function Body */
    *info = 0;
    if (*n < 0) {
	*info = -1;
	i__1 = -(*info);
	xerbla_("ZPTTRF", &i__1);
	return;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return;
    }

/*     Compute the L*D*L**H (or U**H *D*U) factorization of A. */

    i4 = (*n - 1) % 4;
    i__1 = i4;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if (d__[i__] <= 0.) {
	    *info = i__;
	    goto L30;
	}
	i__2 = i__;
	eir = e[i__2].r;
	eii = d_imag(&e[i__]);
	f = eir / d__[i__];
	g = eii / d__[i__];
	i__2 = i__;
	z__1.r = f, z__1.i = g;
	e[i__2].r = z__1.r, e[i__2].i = z__1.i;
	d__[i__ + 1] = d__[i__ + 1] - f * eir - g * eii;
/* L10: */
    }

    i__1 = *n - 4;
    for (i__ = i4 + 1; i__ <= i__1; i__ += 4) {

/*        Drop out of the loop if d(i) <= 0: the matrix is not positive */
/*        definite. */

	if (d__[i__] <= 0.) {
	    *info = i__;
	    goto L30;
	}

/*        Solve for e(i) and d(i+1). */

	i__2 = i__;
	eir = e[i__2].r;
	eii = d_imag(&e[i__]);
	f = eir / d__[i__];
	g = eii / d__[i__];
	i__2 = i__;
	z__1.r = f, z__1.i = g;
	e[i__2].r = z__1.r, e[i__2].i = z__1.i;
	d__[i__ + 1] = d__[i__ + 1] - f * eir - g * eii;

	if (d__[i__ + 1] <= 0.) {
	    *info = i__ + 1;
	    goto L30;
	}

/*        Solve for e(i+1) and d(i+2). */

	i__2 = i__ + 1;
	eir = e[i__2].r;
	eii = d_imag(&e[i__ + 1]);
	f = eir / d__[i__ + 1];
	g = eii / d__[i__ + 1];
	i__2 = i__ + 1;
	z__1.r = f, z__1.i = g;
	e[i__2].r = z__1.r, e[i__2].i = z__1.i;
	d__[i__ + 2] = d__[i__ + 2] - f * eir - g * eii;

	if (d__[i__ + 2] <= 0.) {
	    *info = i__ + 2;
	    goto L30;
	}

/*        Solve for e(i+2) and d(i+3). */

	i__2 = i__ + 2;
	eir = e[i__2].r;
	eii = d_imag(&e[i__ + 2]);
	f = eir / d__[i__ + 2];
	g = eii / d__[i__ + 2];
	i__2 = i__ + 2;
	z__1.r = f, z__1.i = g;
	e[i__2].r = z__1.r, e[i__2].i = z__1.i;
	d__[i__ + 3] = d__[i__ + 3] - f * eir - g * eii;

	if (d__[i__ + 3] <= 0.) {
	    *info = i__ + 3;
	    goto L30;
	}

/*        Solve for e(i+3) and d(i+4). */

	i__2 = i__ + 3;
	eir = e[i__2].r;
	eii = d_imag(&e[i__ + 3]);
	f = eir / d__[i__ + 3];
	g = eii / d__[i__ + 3];
	i__2 = i__ + 3;
	z__1.r = f, z__1.i = g;
	e[i__2].r = z__1.r, e[i__2].i = z__1.i;
	d__[i__ + 4] = d__[i__ + 4] - f * eir - g * eii;
/* L20: */
    }

/*     Check d(n) for positive definiteness. */

    if (d__[*n] <= 0.) {
	*info = *n;
    }

L30:
    return;

/*     End of ZPTTRF */

} /* wpttrf_ */

