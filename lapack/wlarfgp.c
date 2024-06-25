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

static quadcomplex c_b5 = {1.,0.};

/* > \brief \b ZLARFGP generates an elementary reflector (Householder matrix) with non-negative beta. */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download ZLARFGP + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/zlarfgp
.f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/zlarfgp
.f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/zlarfgp
.f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE ZLARFGP( N, ALPHA, X, INCX, TAU ) */

/*       INTEGER            INCX, N */
/*       COMPLEX*16         ALPHA, TAU */
/*       COMPLEX*16         X( * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > ZLARFGP generates a complex elementary reflector H of order n, such */
/* > that */
/* > */
/* >       H**H * ( alpha ) = ( beta ),   H**H * H = I. */
/* >              (   x   )   (   0  ) */
/* > */
/* > where alpha and beta are scalars, beta is doublereal and non-negative, and */
/* > x is an (n-1)-element complex vector.  H is represented in the form */
/* > */
/* >       H = I - tau * ( 1 ) * ( 1 v**H ) , */
/* >                     ( v ) */
/* > */
/* > where tau is a complex scalar and v is a complex (n-1)-element */
/* > vector. Note that H is not hermitian. */
/* > */
/* > If the elements of x are all zero and alpha is doublereal, then tau = 0 */
/* > and H is taken to be the unit matrix. */
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
void  wlarfgp_(integer *n, quadcomplex *alpha, quadcomplex 
	*x, integer *incx, quadcomplex *tau)
{
    /* System generated locals */
    integer i__1, i__2;
    quadreal d__1, d__2;
    quadcomplex z__1, z__2;

    /* Local variables */
    integer j;
    quadcomplex savealpha;
    integer knt;
    quadreal beta, alphi, alphr;
    extern void  wscal_(integer *, quadcomplex *, 
	    quadcomplex *, integer *);
    quadreal xnorm;
    extern quadreal qlapy2_(quadreal *, quadreal *), qlapy3_(quadreal 
	    *, quadreal *, quadreal *), qwnrm2_(integer *, quadcomplex *
	    , integer *), qlamch_(char *);
    extern void  wqscal_(integer *, quadreal *, 
	    quadcomplex *, integer *);
    quadreal bignum;
    extern /* Double Complex */ VOID wladiv_(quadcomplex *, quadcomplex *,
	     quadcomplex *);
    quadreal smlnum;


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
    xnorm = qwnrm2_(&i__1, &x[1], incx);
    alphr = alpha->r;
    alphi = d_imag(alpha);

    if (xnorm == 0.) {

/*        H  =  [1-alpha/abs(alpha) 0; 0 I], sign chosen so ALPHA >= 0. */

	if (alphi == 0.) {
	    if (alphr >= 0.) {
/*              When TAU.eq.ZERO, the vector is special-cased to be */
/*              all zeros in the application routines.  We do not need */
/*              to clear it. */
		tau->r = 0., tau->i = 0.;
	    } else {
/*              However, the application routines rely on explicit */
/*              zero checks when TAU.ne.ZERO, and we must clear X. */
		tau->r = 2., tau->i = 0.;
		i__1 = *n - 1;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = (j - 1) * *incx + 1;
		    x[i__2].r = 0., x[i__2].i = 0.;
		}
		z__1.r = -alpha->r, z__1.i = -alpha->i;
		alpha->r = z__1.r, alpha->i = z__1.i;
	    }
	} else {
/*           Only "reflecting" the diagonal entry to be doublereal and non-negative. */
	    xnorm = qlapy2_(&alphr, &alphi);
	    d__1 = 1. - alphr / xnorm;
	    d__2 = -alphi / xnorm;
	    z__1.r = d__1, z__1.i = d__2;
	    tau->r = z__1.r, tau->i = z__1.i;
	    i__1 = *n - 1;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = (j - 1) * *incx + 1;
		x[i__2].r = 0., x[i__2].i = 0.;
	    }
	    alpha->r = xnorm, alpha->i = 0.;
	}
    } else {

/*        general case */

	d__1 = qlapy3_(&alphr, &alphi, &xnorm);
	beta = d_sign(&d__1, &alphr);
	smlnum = qlamch_("S") / qlamch_("E");
	bignum = 1. / smlnum;

	knt = 0;
	if (abs(beta) < smlnum) {

/*           XNORM, BETA may be inaccurate; scale X and recompute them */

L10:
	    ++knt;
	    i__1 = *n - 1;
	    wqscal_(&i__1, &bignum, &x[1], incx);
	    beta *= bignum;
	    alphi *= bignum;
	    alphr *= bignum;
	    if (abs(beta) < smlnum && knt < 20) {
		goto L10;
	    }

/*           New BETA is at most 1, at least SMLNUM */

	    i__1 = *n - 1;
	    xnorm = qwnrm2_(&i__1, &x[1], incx);
	    z__1.r = alphr, z__1.i = alphi;
	    alpha->r = z__1.r, alpha->i = z__1.i;
	    d__1 = qlapy3_(&alphr, &alphi, &xnorm);
	    beta = d_sign(&d__1, &alphr);
	}
	savealpha.r = alpha->r, savealpha.i = alpha->i;
	z__1.r = alpha->r + beta, z__1.i = alpha->i;
	alpha->r = z__1.r, alpha->i = z__1.i;
	if (beta < 0.) {
	    beta = -beta;
	    z__2.r = -alpha->r, z__2.i = -alpha->i;
	    z__1.r = z__2.r / beta, z__1.i = z__2.i / beta;
	    tau->r = z__1.r, tau->i = z__1.i;
	} else {
	    alphr = alphi * (alphi / alpha->r);
	    alphr += xnorm * (xnorm / alpha->r);
	    d__1 = alphr / beta;
	    d__2 = -alphi / beta;
	    z__1.r = d__1, z__1.i = d__2;
	    tau->r = z__1.r, tau->i = z__1.i;
	    d__1 = -alphr;
	    z__1.r = d__1, z__1.i = alphi;
	    alpha->r = z__1.r, alpha->i = z__1.i;
	}
	wladiv_(&z__1, &c_b5, alpha);
	alpha->r = z__1.r, alpha->i = z__1.i;

	if (z_abs(tau) <= smlnum) {

/*           In the case where the computed TAU ends up being a denormalized number, */
/*           it loses relative accuracy. This is a BIG problem. Solution: flush TAU */
/*           to ZERO (or TWO or whatever makes a nonnegative doublereal number for BETA). */

/*           (Bug report provided by Pat Quillen from MathWorks on Jul 29, 2009.) */
/*           (Thanks Pat. Thanks MathWorks.) */

	    alphr = savealpha.r;
	    alphi = d_imag(&savealpha);
	    if (alphi == 0.) {
		if (alphr >= 0.) {
		    tau->r = 0., tau->i = 0.;
		} else {
		    tau->r = 2., tau->i = 0.;
		    i__1 = *n - 1;
		    for (j = 1; j <= i__1; ++j) {
			i__2 = (j - 1) * *incx + 1;
			x[i__2].r = 0., x[i__2].i = 0.;
		    }
		    z__1.r = -savealpha.r, z__1.i = -savealpha.i;
		    beta = z__1.r;
		}
	    } else {
		xnorm = qlapy2_(&alphr, &alphi);
		d__1 = 1. - alphr / xnorm;
		d__2 = -alphi / xnorm;
		z__1.r = d__1, z__1.i = d__2;
		tau->r = z__1.r, tau->i = z__1.i;
		i__1 = *n - 1;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = (j - 1) * *incx + 1;
		    x[i__2].r = 0., x[i__2].i = 0.;
		}
		beta = xnorm;
	    }

	} else {

/*           This is the general case. */

	    i__1 = *n - 1;
	    wscal_(&i__1, alpha, &x[1], incx);

	}

/*        If BETA is subnormal, it may lose relative accuracy */

	i__1 = knt;
	for (j = 1; j <= i__1; ++j) {
	    beta *= smlnum;
/* L20: */
	}
	alpha->r = beta, alpha->i = 0.;
    }

    return;

/*     End of ZLARFGP */

} /* wlarfgp_ */

