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

/* > \brief \b ZLAIC1 applies one step of incremental condition estimation. */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download ZLAIC1 + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/zlaic1.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/zlaic1.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/zlaic1.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE ZLAIC1( JOB, J, X, SEST, W, GAMMA, SESTPR, S, C ) */

/*       INTEGER            J, JOB */
/*       DOUBLE PRECISION   SEST, SESTPR */
/*       COMPLEX*16         C, GAMMA, S */
/*       COMPLEX*16         W( J ), X( J ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > ZLAIC1 applies one step of incremental condition estimation in */
/* > its simplest version: */
/* > */
/* > Let x, twonorm(x) = 1, be an approximate singular vector of an j-by-j */
/* > lower triangular matrix L, such that */
/* >          twonorm(L*x) = sest */
/* > Then ZLAIC1 computes sestpr, s, c such that */
/* > the vector */
/* >                 [ s*x ] */
/* >          xhat = [  c  ] */
/* > is an approximate singular vector of */
/* >                 [ L       0  ] */
/* >          Lhat = [ w**H gamma ] */
/* > in the sense that */
/* >          twonorm(Lhat*xhat) = sestpr. */
/* > */
/* > Depending on JOB, an estimate for the largest or smallest singular */
/* > value is computed. */
/* > */
/* > Note that [s c]**H and sestpr**2 is an eigenpair of the system */
/* > */
/* >     diag(sest*sest, 0) + [alpha  gamma] * [ conjg(alpha) ] */
/* >                                           [ conjg(gamma) ] */
/* > */
/* > where  alpha =  x**H * w. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] JOB */
/* > \verbatim */
/* >          JOB is INTEGER */
/* >          = 1: an estimate for the largest singular value is computed. */
/* >          = 2: an estimate for the smallest singular value is computed. */
/* > \endverbatim */
/* > */
/* > \param[in] J */
/* > \verbatim */
/* >          J is INTEGER */
/* >          Length of X and W */
/* > \endverbatim */
/* > */
/* > \param[in] X */
/* > \verbatim */
/* >          X is COMPLEX*16 array, dimension (J) */
/* >          The j-vector x. */
/* > \endverbatim */
/* > */
/* > \param[in] SEST */
/* > \verbatim */
/* >          SEST is DOUBLE PRECISION */
/* >          Estimated singular value of j by j matrix L */
/* > \endverbatim */
/* > */
/* > \param[in] W */
/* > \verbatim */
/* >          W is COMPLEX*16 array, dimension (J) */
/* >          The j-vector w. */
/* > \endverbatim */
/* > */
/* > \param[in] GAMMA */
/* > \verbatim */
/* >          GAMMA is COMPLEX*16 */
/* >          The diagonal element gamma. */
/* > \endverbatim */
/* > */
/* > \param[out] SESTPR */
/* > \verbatim */
/* >          SESTPR is DOUBLE PRECISION */
/* >          Estimated singular value of (j+1) by (j+1) matrix Lhat. */
/* > \endverbatim */
/* > */
/* > \param[out] S */
/* > \verbatim */
/* >          S is COMPLEX*16 */
/* >          Sine needed in forming xhat. */
/* > \endverbatim */
/* > */
/* > \param[out] C */
/* > \verbatim */
/* >          C is COMPLEX*16 */
/* >          Cosine needed in forming xhat. */
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
void  wlaic1_(integer *job, integer *j, quadcomplex *x, 
	quadreal *sest, quadcomplex *w, quadcomplex *gamma, quadreal *
	sestpr, quadcomplex *s, quadcomplex *c__)
{
    /* System generated locals */
    quadreal d__1, d__2;
    quadcomplex z__1, z__2, z__3, z__4, z__5, z__6;

    /* Local variables */
    quadreal b, t, s1, s2, scl, eps, tmp;
    quadcomplex sine;
    quadreal test, zeta1, zeta2;
    quadcomplex alpha;
    quadreal norma;
    extern /* Double Complex */ VOID wqotc_(quadcomplex *, integer *, 
	    quadcomplex *, integer *, quadcomplex *, integer *);
    extern quadreal qlamch_(char *);
    quadreal absgam, absalp;
    quadcomplex cosine;
    quadreal absest;


/*  -- LAPACK auxiliary routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     December 2016 */


/*  ===================================================================== */


    /* Parameter adjustments */
    --w;
    --x;

    /* Function Body */
    eps = qlamch_("Epsilon");
    wqotc_(&z__1, j, &x[1], &c__1, &w[1], &c__1);
    alpha.r = z__1.r, alpha.i = z__1.i;

    absalp = z_abs(&alpha);
    absgam = z_abs(gamma);
    absest = abs(*sest);

    if (*job == 1) {

/*        Estimating largest singular value */

/*        special cases */

	if (*sest == 0.) {
	    s1 = f2cmax(absgam,absalp);
	    if (s1 == 0.) {
		s->r = 0., s->i = 0.;
		c__->r = 1., c__->i = 0.;
		*sestpr = 0.;
	    } else {
		z__1.r = alpha.r / s1, z__1.i = alpha.i / s1;
		s->r = z__1.r, s->i = z__1.i;
		z__1.r = gamma->r / s1, z__1.i = gamma->i / s1;
		c__->r = z__1.r, c__->i = z__1.i;
		d_cnjg(&z__4, s);
		z__3.r = s->r * z__4.r - s->i * z__4.i, z__3.i = s->r * 
			z__4.i + s->i * z__4.r;
		d_cnjg(&z__6, c__);
		z__5.r = c__->r * z__6.r - c__->i * z__6.i, z__5.i = c__->r * 
			z__6.i + c__->i * z__6.r;
		z__2.r = z__3.r + z__5.r, z__2.i = z__3.i + z__5.i;
		z_sqrt(&z__1, &z__2);
		tmp = z__1.r;
		z__1.r = s->r / tmp, z__1.i = s->i / tmp;
		s->r = z__1.r, s->i = z__1.i;
		z__1.r = c__->r / tmp, z__1.i = c__->i / tmp;
		c__->r = z__1.r, c__->i = z__1.i;
		*sestpr = s1 * tmp;
	    }
	    return;
	} else if (absgam <= eps * absest) {
	    s->r = 1., s->i = 0.;
	    c__->r = 0., c__->i = 0.;
	    tmp = f2cmax(absest,absalp);
	    s1 = absest / tmp;
	    s2 = absalp / tmp;
	    *sestpr = tmp * M(sqrt)(s1 * s1 + s2 * s2);
	    return;
	} else if (absalp <= eps * absest) {
	    s1 = absgam;
	    s2 = absest;
	    if (s1 <= s2) {
		s->r = 1., s->i = 0.;
		c__->r = 0., c__->i = 0.;
		*sestpr = s2;
	    } else {
		s->r = 0., s->i = 0.;
		c__->r = 1., c__->i = 0.;
		*sestpr = s1;
	    }
	    return;
	} else if (absest <= eps * absalp || absest <= eps * absgam) {
	    s1 = absgam;
	    s2 = absalp;
	    if (s1 <= s2) {
		tmp = s1 / s2;
		scl = M(sqrt)(tmp * tmp + 1.);
		*sestpr = s2 * scl;
		z__2.r = alpha.r / s2, z__2.i = alpha.i / s2;
		z__1.r = z__2.r / scl, z__1.i = z__2.i / scl;
		s->r = z__1.r, s->i = z__1.i;
		z__2.r = gamma->r / s2, z__2.i = gamma->i / s2;
		z__1.r = z__2.r / scl, z__1.i = z__2.i / scl;
		c__->r = z__1.r, c__->i = z__1.i;
	    } else {
		tmp = s2 / s1;
		scl = M(sqrt)(tmp * tmp + 1.);
		*sestpr = s1 * scl;
		z__2.r = alpha.r / s1, z__2.i = alpha.i / s1;
		z__1.r = z__2.r / scl, z__1.i = z__2.i / scl;
		s->r = z__1.r, s->i = z__1.i;
		z__2.r = gamma->r / s1, z__2.i = gamma->i / s1;
		z__1.r = z__2.r / scl, z__1.i = z__2.i / scl;
		c__->r = z__1.r, c__->i = z__1.i;
	    }
	    return;
	} else {

/*           normal case */

	    zeta1 = absalp / absest;
	    zeta2 = absgam / absest;

	    b = (1. - zeta1 * zeta1 - zeta2 * zeta2) * .5;
	    d__1 = zeta1 * zeta1;
	    c__->r = d__1, c__->i = 0.;
	    if (b > 0.) {
		d__1 = b * b;
		z__4.r = d__1 + c__->r, z__4.i = c__->i;
		z_sqrt(&z__3, &z__4);
		z__2.r = b + z__3.r, z__2.i = z__3.i;
		z_div(&z__1, c__, &z__2);
		t = z__1.r;
	    } else {
		d__1 = b * b;
		z__3.r = d__1 + c__->r, z__3.i = c__->i;
		z_sqrt(&z__2, &z__3);
		z__1.r = z__2.r - b, z__1.i = z__2.i;
		t = z__1.r;
	    }

	    z__3.r = alpha.r / absest, z__3.i = alpha.i / absest;
	    z__2.r = -z__3.r, z__2.i = -z__3.i;
	    z__1.r = z__2.r / t, z__1.i = z__2.i / t;
	    sine.r = z__1.r, sine.i = z__1.i;
	    z__3.r = gamma->r / absest, z__3.i = gamma->i / absest;
	    z__2.r = -z__3.r, z__2.i = -z__3.i;
	    d__1 = t + 1.;
	    z__1.r = z__2.r / d__1, z__1.i = z__2.i / d__1;
	    cosine.r = z__1.r, cosine.i = z__1.i;
	    d_cnjg(&z__4, &sine);
	    z__3.r = sine.r * z__4.r - sine.i * z__4.i, z__3.i = sine.r * 
		    z__4.i + sine.i * z__4.r;
	    d_cnjg(&z__6, &cosine);
	    z__5.r = cosine.r * z__6.r - cosine.i * z__6.i, z__5.i = cosine.r 
		    * z__6.i + cosine.i * z__6.r;
	    z__2.r = z__3.r + z__5.r, z__2.i = z__3.i + z__5.i;
	    z_sqrt(&z__1, &z__2);
	    tmp = z__1.r;
	    z__1.r = sine.r / tmp, z__1.i = sine.i / tmp;
	    s->r = z__1.r, s->i = z__1.i;
	    z__1.r = cosine.r / tmp, z__1.i = cosine.i / tmp;
	    c__->r = z__1.r, c__->i = z__1.i;
	    *sestpr = M(sqrt)(t + 1.) * absest;
	    return;
	}

    } else if (*job == 2) {

/*        Estimating smallest singular value */

/*        special cases */

	if (*sest == 0.) {
	    *sestpr = 0.;
	    if (f2cmax(absgam,absalp) == 0.) {
		sine.r = 1., sine.i = 0.;
		cosine.r = 0., cosine.i = 0.;
	    } else {
		d_cnjg(&z__2, gamma);
		z__1.r = -z__2.r, z__1.i = -z__2.i;
		sine.r = z__1.r, sine.i = z__1.i;
		d_cnjg(&z__1, &alpha);
		cosine.r = z__1.r, cosine.i = z__1.i;
	    }
/* Computing MAX */
	    d__1 = z_abs(&sine), d__2 = z_abs(&cosine);
	    s1 = f2cmax(d__1,d__2);
	    z__1.r = sine.r / s1, z__1.i = sine.i / s1;
	    s->r = z__1.r, s->i = z__1.i;
	    z__1.r = cosine.r / s1, z__1.i = cosine.i / s1;
	    c__->r = z__1.r, c__->i = z__1.i;
	    d_cnjg(&z__4, s);
	    z__3.r = s->r * z__4.r - s->i * z__4.i, z__3.i = s->r * z__4.i + 
		    s->i * z__4.r;
	    d_cnjg(&z__6, c__);
	    z__5.r = c__->r * z__6.r - c__->i * z__6.i, z__5.i = c__->r * 
		    z__6.i + c__->i * z__6.r;
	    z__2.r = z__3.r + z__5.r, z__2.i = z__3.i + z__5.i;
	    z_sqrt(&z__1, &z__2);
	    tmp = z__1.r;
	    z__1.r = s->r / tmp, z__1.i = s->i / tmp;
	    s->r = z__1.r, s->i = z__1.i;
	    z__1.r = c__->r / tmp, z__1.i = c__->i / tmp;
	    c__->r = z__1.r, c__->i = z__1.i;
	    return;
	} else if (absgam <= eps * absest) {
	    s->r = 0., s->i = 0.;
	    c__->r = 1., c__->i = 0.;
	    *sestpr = absgam;
	    return;
	} else if (absalp <= eps * absest) {
	    s1 = absgam;
	    s2 = absest;
	    if (s1 <= s2) {
		s->r = 0., s->i = 0.;
		c__->r = 1., c__->i = 0.;
		*sestpr = s1;
	    } else {
		s->r = 1., s->i = 0.;
		c__->r = 0., c__->i = 0.;
		*sestpr = s2;
	    }
	    return;
	} else if (absest <= eps * absalp || absest <= eps * absgam) {
	    s1 = absgam;
	    s2 = absalp;
	    if (s1 <= s2) {
		tmp = s1 / s2;
		scl = M(sqrt)(tmp * tmp + 1.);
		*sestpr = absest * (tmp / scl);
		d_cnjg(&z__4, gamma);
		z__3.r = z__4.r / s2, z__3.i = z__4.i / s2;
		z__2.r = -z__3.r, z__2.i = -z__3.i;
		z__1.r = z__2.r / scl, z__1.i = z__2.i / scl;
		s->r = z__1.r, s->i = z__1.i;
		d_cnjg(&z__3, &alpha);
		z__2.r = z__3.r / s2, z__2.i = z__3.i / s2;
		z__1.r = z__2.r / scl, z__1.i = z__2.i / scl;
		c__->r = z__1.r, c__->i = z__1.i;
	    } else {
		tmp = s2 / s1;
		scl = M(sqrt)(tmp * tmp + 1.);
		*sestpr = absest / scl;
		d_cnjg(&z__4, gamma);
		z__3.r = z__4.r / s1, z__3.i = z__4.i / s1;
		z__2.r = -z__3.r, z__2.i = -z__3.i;
		z__1.r = z__2.r / scl, z__1.i = z__2.i / scl;
		s->r = z__1.r, s->i = z__1.i;
		d_cnjg(&z__3, &alpha);
		z__2.r = z__3.r / s1, z__2.i = z__3.i / s1;
		z__1.r = z__2.r / scl, z__1.i = z__2.i / scl;
		c__->r = z__1.r, c__->i = z__1.i;
	    }
	    return;
	} else {

/*           normal case */

	    zeta1 = absalp / absest;
	    zeta2 = absgam / absest;

/* Computing MAX */
	    d__1 = zeta1 * zeta1 + 1. + zeta1 * zeta2, d__2 = zeta1 * zeta2 + 
		    zeta2 * zeta2;
	    norma = f2cmax(d__1,d__2);

/*           See if root is closer to zero or to ONE */

	    test = (zeta1 - zeta2) * 2. * (zeta1 + zeta2) + 1.;
	    if (test >= 0.) {

/*              root is close to zero, compute directly */

		b = (zeta1 * zeta1 + zeta2 * zeta2 + 1.) * .5;
		d__1 = zeta2 * zeta2;
		c__->r = d__1, c__->i = 0.;
		d__2 = b * b;
		z__2.r = d__2 - c__->r, z__2.i = -c__->i;
		d__1 = b + M(sqrt)(z_abs(&z__2));
		z__1.r = c__->r / d__1, z__1.i = c__->i / d__1;
		t = z__1.r;
		z__2.r = alpha.r / absest, z__2.i = alpha.i / absest;
		d__1 = 1. - t;
		z__1.r = z__2.r / d__1, z__1.i = z__2.i / d__1;
		sine.r = z__1.r, sine.i = z__1.i;
		z__3.r = gamma->r / absest, z__3.i = gamma->i / absest;
		z__2.r = -z__3.r, z__2.i = -z__3.i;
		z__1.r = z__2.r / t, z__1.i = z__2.i / t;
		cosine.r = z__1.r, cosine.i = z__1.i;
		*sestpr = M(sqrt)(t + eps * 4. * eps * norma) * absest;
	    } else {

/*              root is closer to ONE, shift by that amount */

		b = (zeta2 * zeta2 + zeta1 * zeta1 - 1.) * .5;
		d__1 = zeta1 * zeta1;
		c__->r = d__1, c__->i = 0.;
		if (b >= 0.) {
		    z__2.r = -c__->r, z__2.i = -c__->i;
		    d__1 = b * b;
		    z__5.r = d__1 + c__->r, z__5.i = c__->i;
		    z_sqrt(&z__4, &z__5);
		    z__3.r = b + z__4.r, z__3.i = z__4.i;
		    z_div(&z__1, &z__2, &z__3);
		    t = z__1.r;
		} else {
		    d__1 = b * b;
		    z__3.r = d__1 + c__->r, z__3.i = c__->i;
		    z_sqrt(&z__2, &z__3);
		    z__1.r = b - z__2.r, z__1.i = -z__2.i;
		    t = z__1.r;
		}
		z__3.r = alpha.r / absest, z__3.i = alpha.i / absest;
		z__2.r = -z__3.r, z__2.i = -z__3.i;
		z__1.r = z__2.r / t, z__1.i = z__2.i / t;
		sine.r = z__1.r, sine.i = z__1.i;
		z__3.r = gamma->r / absest, z__3.i = gamma->i / absest;
		z__2.r = -z__3.r, z__2.i = -z__3.i;
		d__1 = t + 1.;
		z__1.r = z__2.r / d__1, z__1.i = z__2.i / d__1;
		cosine.r = z__1.r, cosine.i = z__1.i;
		*sestpr = M(sqrt)(t + 1. + eps * 4. * eps * norma) * absest;
	    }
	    d_cnjg(&z__4, &sine);
	    z__3.r = sine.r * z__4.r - sine.i * z__4.i, z__3.i = sine.r * 
		    z__4.i + sine.i * z__4.r;
	    d_cnjg(&z__6, &cosine);
	    z__5.r = cosine.r * z__6.r - cosine.i * z__6.i, z__5.i = cosine.r 
		    * z__6.i + cosine.i * z__6.r;
	    z__2.r = z__3.r + z__5.r, z__2.i = z__3.i + z__5.i;
	    z_sqrt(&z__1, &z__2);
	    tmp = z__1.r;
	    z__1.r = sine.r / tmp, z__1.i = sine.i / tmp;
	    s->r = z__1.r, s->i = z__1.i;
	    z__1.r = cosine.r / tmp, z__1.i = cosine.i / tmp;
	    c__->r = z__1.r, c__->i = z__1.i;
	    return;

	}
    }
    return;

/*     End of ZLAIC1 */

} /* wlaic1_ */

