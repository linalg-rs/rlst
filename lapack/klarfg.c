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

/* Table of constant values */

static halfcomplex c_b5 = {1.,0.};

/* > \brief \b ZLARFG generates an elementary reflector (Householder matrix). */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download ZLARFG + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/zlarfg.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/zlarfg.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/zlarfg.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE ZLARFG( N, ALPHA, X, INCX, TAU ) */

/*       INTEGER            INCX, N */
/*       COMPLEX*16         ALPHA, TAU */
/*       COMPLEX*16         X( * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > ZLARFG generates a complex elementary reflector H of order n, such */
/* > that */
/* > */
/* >       H**H * ( alpha ) = ( beta ),   H**H * H = I. */
/* >              (   x   )   (   0  ) */
/* > */
/* > where alpha and beta are scalars, with beta doublereal, and x is an */
/* > (n-1)-element complex vector. H is represented in the form */
/* > */
/* >       H = I - tau * ( 1 ) * ( 1 v**H ) , */
/* >                     ( v ) */
/* > */
/* > where tau is a complex scalar and v is a complex (n-1)-element */
/* > vector. Note that H is not hermitian. */
/* > */
/* > If the elements of x are all zero and alpha is doublereal, then tau = 0 */
/* > and H is taken to be the unit matrix. */
/* > */
/* > Otherwise  1 <= doublereal(tau) <= 2  and  abs(tau-1) <= 1 . */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >          The order of the elementary reflector. */
/* > \endverbatim */
/* > */
/* > \param[in,out] ALPHA */
/* > \verbatim */
/* >          ALPHA is COMPLEX*16 */
/* >          On entry, the value alpha. */
/* >          On exit, it is overwritten with the value beta. */
/* > \endverbatim */
/* > */
/* > \param[in,out] X */
/* > \verbatim */
/* >          X is COMPLEX*16 array, dimension */
/* >                         (1+(N-2)*abs(INCX)) */
/* >          On entry, the vector x. */
/* >          On exit, it is overwritten with the vector v. */
/* > \endverbatim */
/* > */
/* > \param[in] INCX */
/* > \verbatim */
/* >          INCX is INTEGER */
/* >          The increment between elements of X. INCX > 0. */
/* > \endverbatim */
/* > */
/* > \param[out] TAU */
/* > \verbatim */
/* >          TAU is COMPLEX*16 */
/* >          The value tau. */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date November 2017 */

/* > \ingroup complex16OTHERauxiliary */

/*  ===================================================================== */
void  klarfg_(integer *n, halfcomplex *alpha, halfcomplex *
	x, integer *incx, halfcomplex *tau)
{
    /* System generated locals */
    integer i__1;
    halfreal d__1, d__2;
    halfcomplex z__1, z__2;

    /* Local variables */
    integer j, knt;
    halfreal beta, alphi, alphr;
    extern void  kscal_(integer *, halfcomplex *, 
	    halfcomplex *, integer *);
    halfreal xnorm;
    extern halfreal hlapy3_(halfreal *, halfreal *, halfreal *), 
	    hwnrm2_(integer *, halfcomplex *, integer *), hlamch_(char *);
    halfreal safmin;
    extern void  whscal_(integer *, halfreal *, 
	    halfcomplex *, integer *);
    halfreal rsafmn;
    extern /* Double Complex */ VOID kladiv_(halfcomplex *, halfcomplex *,
	     halfcomplex *);


/*  -- LAPACK auxiliary routine (version 3.8.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     November 2017 */


/*  ===================================================================== */


    /* Parameter adjustments */
    --x;

    /* Function Body */
    if (*n <= 0) {
	tau->r = 0., tau->i = 0.;
	return;
    }

    i__1 = *n - 1;
    xnorm = hwnrm2_(&i__1, &x[1], incx);
    alphr = alpha->r;
    alphi = d_imag(alpha);

    if (xnorm == 0. && alphi == 0.) {

/*        H  =  I */

	tau->r = 0., tau->i = 0.;
    } else {

/*        general case */

	d__1 = hlapy3_(&alphr, &alphi, &xnorm);
	beta = -d_sign(&d__1, &alphr);
	safmin = hlamch_("S") / hlamch_("E");
	rsafmn = 1. / safmin;

	knt = 0;
	if (abs(beta) < safmin) {

/*           XNORM, BETA may be inaccurate; scale X and recompute them */

L10:
	    ++knt;
	    i__1 = *n - 1;
	    whscal_(&i__1, &rsafmn, &x[1], incx);
	    beta *= rsafmn;
	    alphi *= rsafmn;
	    alphr *= rsafmn;
	    if (abs(beta) < safmin && knt < 20) {
		goto L10;
	    }

/*           New BETA is at most 1, at least SAFMIN */

	    i__1 = *n - 1;
	    xnorm = hwnrm2_(&i__1, &x[1], incx);
	    z__1.r = alphr, z__1.i = alphi;
	    alpha->r = z__1.r, alpha->i = z__1.i;
	    d__1 = hlapy3_(&alphr, &alphi, &xnorm);
	    beta = -d_sign(&d__1, &alphr);
	}
	d__1 = (beta - alphr) / beta;
	d__2 = -alphi / beta;
	z__1.r = d__1, z__1.i = d__2;
	tau->r = z__1.r, tau->i = z__1.i;
	z__2.r = alpha->r - beta, z__2.i = alpha->i;
	kladiv_(&z__1, &c_b5, &z__2);
	alpha->r = z__1.r, alpha->i = z__1.i;
	i__1 = *n - 1;
	kscal_(&i__1, alpha, &x[1], incx);

/*        If ALPHA is subnormal, it may lose relative accuracy */

	i__1 = knt;
	for (j = 1; j <= i__1; ++j) {
	    beta *= safmin;
/* L20: */
	}
	alpha->r = beta, alpha->i = 0.;
    }

    return;

/*     End of ZLARFG */

} /* klarfg_ */

