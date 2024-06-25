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

/* Table of constant values */

static integer c__1 = 1;

/* > \brief \b ZLACON estimates the 1-norm of a square matrix, using reverse communication for evaluating matr
ix-vector products. */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download ZLACON + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/zlacon.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/zlacon.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/zlacon.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE ZLACON( N, V, X, EST, KASE ) */

/*       INTEGER            KASE, N */
/*       DOUBLE PRECISION   EST */
/*       COMPLEX*16         V( N ), X( N ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > ZLACON estimates the 1-norm of a square, complex matrix A. */
/* > Reverse communication is used for evaluating matrix-vector products. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >         The order of the matrix.  N >= 1. */
/* > \endverbatim */
/* > */
/* > \param[out] V */
/* > \verbatim */
/* >          V is COMPLEX*16 array, dimension (N) */
/* >         On the final return, V = A*W,  where  EST = norm(V)/norm(W) */
/* >         (W is not returned). */
/* > \endverbatim */
/* > */
/* > \param[in,out] X */
/* > \verbatim */
/* >          X is COMPLEX*16 array, dimension (N) */
/* >         On an intermediate return, X should be overwritten by */
/* >               A * X,   if KASE=1, */
/* >               A**H * X,  if KASE=2, */
/* >         where A**H is the conjugate transpose of A, and ZLACON must be */
/* >         re-called with all the other parameters unchanged. */
/* > \endverbatim */
/* > */
/* > \param[in,out] EST */
/* > \verbatim */
/* >          EST is DOUBLE PRECISION */
/* >         On entry with KASE = 1 or 2 and JUMP = 3, EST should be */
/* >         unchanged from the previous call to ZLACON. */
/* >         On exit, EST is an estimate (a lower bound) for norm(A). */
/* > \endverbatim */
/* > */
/* > \param[in,out] KASE */
/* > \verbatim */
/* >          KASE is INTEGER */
/* >         On the initial call to ZLACON, KASE should be 0. */
/* >         On an intermediate return, KASE will be 1 or 2, indicating */
/* >         whether X should be overwritten by A * X  or A**H * X. */
/* >         On the final return from ZLACON, KASE will again be 0. */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date December 2016 */

/* > \ingroup complex16OTHERauxiliary */

/* > \par Further Details: */
/*  ===================== */
/* > */
/* >  Originally named CONEST, dated March 16, 1988. \n */
/* >  Last modified:  April, 1999 */

/* > \par Contributors: */
/*  ================== */
/* > */
/* >     Nick Higham, University of Manchester */

/* > \par References: */
/*  ================ */
/* > */
/* >  N.J. Higham, "FORTRAN codes for estimating the one-norm of */
/* >  a doublereal or complex matrix, with applications to condition estimation", */
/* >  ACM Trans. Math. Soft., vol. 14, no. 4, pp. 381-396, December 1988. */
/* > */
/*  ===================================================================== */
void  wlacon_(integer *n, quadcomplex *v, quadcomplex *x, 
	quadreal *est, integer *kase)
{
    /* System generated locals */
    integer i__1, i__2, i__3;
    quadreal d__1, d__2;
    quadcomplex z__1;

    /* Local variables */
    static integer i__, j, iter;
    static quadreal temp;
    static integer jump;
    static quadreal absxi;
    static integer jlast;
    extern void  wcopy_(integer *, quadcomplex *, integer *, 
	    quadcomplex *, integer *);
    extern integer iwmax1_(integer *, quadcomplex *, integer *);
    extern quadreal qwsum1_(integer *, quadcomplex *, integer *), qlamch_(
	    char *);
    static quadreal safmin, altsgn, estold;


/*  -- LAPACK auxiliary routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     December 2016 */


/*  ===================================================================== */


    /* Parameter adjustments */
    --x;
    --v;

    /* Function Body */
    safmin = qlamch_("Safe minimum");
    if (*kase == 0) {
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    i__2 = i__;
	    d__1 = 1. / (quadreal) (*n);
	    z__1.r = d__1, z__1.i = 0.;
	    x[i__2].r = z__1.r, x[i__2].i = z__1.i;
/* L10: */
	}
	*kase = 1;
	jump = 1;
	return;
    }

    switch (jump) {
	case 1:  goto L20;
	case 2:  goto L40;
	case 3:  goto L70;
	case 4:  goto L90;
	case 5:  goto L120;
    }

/*     ................ ENTRY   (JUMP = 1) */
/*     FIRST ITERATION.  X HAS BEEN OVERWRITTEN BY A*X. */

L20:
    if (*n == 1) {
	v[1].r = x[1].r, v[1].i = x[1].i;
	*est = z_abs(&v[1]);
/*        ... QUIT */
	goto L130;
    }
    *est = qwsum1_(n, &x[1], &c__1);

    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	absxi = z_abs(&x[i__]);
	if (absxi > safmin) {
	    i__2 = i__;
	    i__3 = i__;
	    d__1 = x[i__3].r / absxi;
	    d__2 = d_imag(&x[i__]) / absxi;
	    z__1.r = d__1, z__1.i = d__2;
	    x[i__2].r = z__1.r, x[i__2].i = z__1.i;
	} else {
	    i__2 = i__;
	    x[i__2].r = 1., x[i__2].i = 0.;
	}
/* L30: */
    }
    *kase = 2;
    jump = 2;
    return;

/*     ................ ENTRY   (JUMP = 2) */
/*     FIRST ITERATION.  X HAS BEEN OVERWRITTEN BY CTRANS(A)*X. */

L40:
    j = iwmax1_(n, &x[1], &c__1);
    iter = 2;

/*     MAIN LOOP - ITERATIONS 2,3,...,ITMAX. */

L50:
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	i__2 = i__;
	x[i__2].r = 0., x[i__2].i = 0.;
/* L60: */
    }
    i__1 = j;
    x[i__1].r = 1., x[i__1].i = 0.;
    *kase = 1;
    jump = 3;
    return;

/*     ................ ENTRY   (JUMP = 3) */
/*     X HAS BEEN OVERWRITTEN BY A*X. */

L70:
    wcopy_(n, &x[1], &c__1, &v[1], &c__1);
    estold = *est;
    *est = qwsum1_(n, &v[1], &c__1);

/*     TEST FOR CYCLING. */
    if (*est <= estold) {
	goto L100;
    }

    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	absxi = z_abs(&x[i__]);
	if (absxi > safmin) {
	    i__2 = i__;
	    i__3 = i__;
	    d__1 = x[i__3].r / absxi;
	    d__2 = d_imag(&x[i__]) / absxi;
	    z__1.r = d__1, z__1.i = d__2;
	    x[i__2].r = z__1.r, x[i__2].i = z__1.i;
	} else {
	    i__2 = i__;
	    x[i__2].r = 1., x[i__2].i = 0.;
	}
/* L80: */
    }
    *kase = 2;
    jump = 4;
    return;

/*     ................ ENTRY   (JUMP = 4) */
/*     X HAS BEEN OVERWRITTEN BY CTRANS(A)*X. */

L90:
    jlast = j;
    j = iwmax1_(n, &x[1], &c__1);
    if (z_abs(&x[jlast]) != z_abs(&x[j]) && iter < 5) {
	++iter;
	goto L50;
    }

/*     ITERATION COMPLETE.  FINAL STAGE. */

L100:
    altsgn = 1.;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	i__2 = i__;
	d__1 = altsgn * ((quadreal) (i__ - 1) / (quadreal) (*n - 1) + 1.);
	z__1.r = d__1, z__1.i = 0.;
	x[i__2].r = z__1.r, x[i__2].i = z__1.i;
	altsgn = -altsgn;
/* L110: */
    }
    *kase = 1;
    jump = 5;
    return;

/*     ................ ENTRY   (JUMP = 5) */
/*     X HAS BEEN OVERWRITTEN BY A*X. */

L120:
    temp = qwsum1_(n, &x[1], &c__1) / (quadreal) (*n * 3) * 2.;
    if (temp > *est) {
	wcopy_(n, &x[1], &c__1, &v[1], &c__1);
	*est = temp;
    }

L130:
    *kase = 0;
    return;

/*     End of ZLACON */

} /* wlacon_ */

