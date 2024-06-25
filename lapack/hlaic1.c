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

static integer c__1 = 1;
static halfreal c_b5 = 1.;

/* > \brief \b DLAIC1 applies one step of incremental condition estimation. */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download DLAIC1 + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlaic1.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlaic1.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlaic1.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE DLAIC1( JOB, J, X, SEST, W, GAMMA, SESTPR, S, C ) */

/*       INTEGER            J, JOB */
/*       DOUBLE PRECISION   C, GAMMA, S, SEST, SESTPR */
/*       DOUBLE PRECISION   W( J ), X( J ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > DLAIC1 applies one step of incremental condition estimation in */
/* > its simplest version: */
/* > */
/* > Let x, twonorm(x) = 1, be an approximate singular vector of an j-by-j */
/* > lower triangular matrix L, such that */
/* >          twonorm(L*x) = sest */
/* > Then DLAIC1 computes sestpr, s, c such that */
/* > the vector */
/* >                 [ s*x ] */
/* >          xhat = [  c  ] */
/* > is an approximate singular vector of */
/* >                 [ L       0  ] */
/* >          Lhat = [ w**T gamma ] */
/* > in the sense that */
/* >          twonorm(Lhat*xhat) = sestpr. */
/* > */
/* > Depending on JOB, an estimate for the largest or smallest singular */
/* > value is computed. */
/* > */
/* > Note that [s c]**T and sestpr**2 is an eigenpair of the system */
/* > */
/* >     diag(sest*sest, 0) + [alpha  gamma] * [ alpha ] */
/* >                                           [ gamma ] */
/* > */
/* > where  alpha =  x**T*w. */
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
/* >          X is DOUBLE PRECISION array, dimension (J) */
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
/* >          W is DOUBLE PRECISION array, dimension (J) */
/* >          The j-vector w. */
/* > \endverbatim */
/* > */
/* > \param[in] GAMMA */
/* > \verbatim */
/* >          GAMMA is DOUBLE PRECISION */
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
/* >          S is DOUBLE PRECISION */
/* >          Sine needed in forming xhat. */
/* > \endverbatim */
/* > */
/* > \param[out] C */
/* > \verbatim */
/* >          C is DOUBLE PRECISION */
/* >          Cosine needed in forming xhat. */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date December 2016 */

/* > \ingroup doubleOTHERauxiliary */

/*  ===================================================================== */
void  hlaic1_(integer *job, integer *j, halfreal *x, 
	halfreal *sest, halfreal *w, halfreal *gamma, halfreal *
	sestpr, halfreal *s, halfreal *c__)
{
    /* System generated locals */
    halfreal d__1, d__2, d__3, d__4;

    /* Local variables */
    halfreal b, t, s1, s2, eps, tmp;
    extern halfreal hdot_(integer *, halfreal *, integer *, halfreal *, 
	    integer *);
    halfreal sine, test, zeta1, zeta2, alpha, norma;
    extern halfreal hlamch_(char *);
    halfreal absgam, absalp, cosine, absest;


/*  -- LAPACK auxiliary routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     December 2016 */


/*  ===================================================================== */


    /* Parameter adjustments */
    --w;
    --x;

    /* Function Body */
    eps = hlamch_("Epsilon");
    alpha = hdot_(j, &x[1], &c__1, &w[1], &c__1);

    absalp = abs(alpha);
    absgam = abs(*gamma);
    absest = abs(*sest);

    if (*job == 1) {

/*        Estimating largest singular value */

/*        special cases */

	if (*sest == 0.) {
	    s1 = f2cmax(absgam,absalp);
	    if (s1 == 0.) {
		*s = 0.;
		*c__ = 1.;
		*sestpr = 0.;
	    } else {
		*s = alpha / s1;
		*c__ = *gamma / s1;
		tmp = M(sqrt)(*s * *s + *c__ * *c__);
		*s /= tmp;
		*c__ /= tmp;
		*sestpr = s1 * tmp;
	    }
	    return;
	} else if (absgam <= eps * absest) {
	    *s = 1.;
	    *c__ = 0.;
	    tmp = f2cmax(absest,absalp);
	    s1 = absest / tmp;
	    s2 = absalp / tmp;
	    *sestpr = tmp * M(sqrt)(s1 * s1 + s2 * s2);
	    return;
	} else if (absalp <= eps * absest) {
	    s1 = absgam;
	    s2 = absest;
	    if (s1 <= s2) {
		*s = 1.;
		*c__ = 0.;
		*sestpr = s2;
	    } else {
		*s = 0.;
		*c__ = 1.;
		*sestpr = s1;
	    }
	    return;
	} else if (absest <= eps * absalp || absest <= eps * absgam) {
	    s1 = absgam;
	    s2 = absalp;
	    if (s1 <= s2) {
		tmp = s1 / s2;
		*s = M(sqrt)(tmp * tmp + 1.);
		*sestpr = s2 * *s;
		*c__ = *gamma / s2 / *s;
		*s = d_sign(&c_b5, &alpha) / *s;
	    } else {
		tmp = s2 / s1;
		*c__ = M(sqrt)(tmp * tmp + 1.);
		*sestpr = s1 * *c__;
		*s = alpha / s1 / *c__;
		*c__ = d_sign(&c_b5, gamma) / *c__;
	    }
	    return;
	} else {

/*           normal case */

	    zeta1 = alpha / absest;
	    zeta2 = *gamma / absest;

	    b = (1. - zeta1 * zeta1 - zeta2 * zeta2) * .5;
	    *c__ = zeta1 * zeta1;
	    if (b > 0.) {
		t = *c__ / (b + M(sqrt)(b * b + *c__));
	    } else {
		t = M(sqrt)(b * b + *c__) - b;
	    }

	    sine = -zeta1 / t;
	    cosine = -zeta2 / (t + 1.);
	    tmp = M(sqrt)(sine * sine + cosine * cosine);
	    *s = sine / tmp;
	    *c__ = cosine / tmp;
	    *sestpr = M(sqrt)(t + 1.) * absest;
	    return;
	}

    } else if (*job == 2) {

/*        Estimating smallest singular value */

/*        special cases */

	if (*sest == 0.) {
	    *sestpr = 0.;
	    if (f2cmax(absgam,absalp) == 0.) {
		sine = 1.;
		cosine = 0.;
	    } else {
		sine = -(*gamma);
		cosine = alpha;
	    }
/* Computing MAX */
	    d__1 = abs(sine), d__2 = abs(cosine);
	    s1 = f2cmax(d__1,d__2);
	    *s = sine / s1;
	    *c__ = cosine / s1;
	    tmp = M(sqrt)(*s * *s + *c__ * *c__);
	    *s /= tmp;
	    *c__ /= tmp;
	    return;
	} else if (absgam <= eps * absest) {
	    *s = 0.;
	    *c__ = 1.;
	    *sestpr = absgam;
	    return;
	} else if (absalp <= eps * absest) {
	    s1 = absgam;
	    s2 = absest;
	    if (s1 <= s2) {
		*s = 0.;
		*c__ = 1.;
		*sestpr = s1;
	    } else {
		*s = 1.;
		*c__ = 0.;
		*sestpr = s2;
	    }
	    return;
	} else if (absest <= eps * absalp || absest <= eps * absgam) {
	    s1 = absgam;
	    s2 = absalp;
	    if (s1 <= s2) {
		tmp = s1 / s2;
		*c__ = M(sqrt)(tmp * tmp + 1.);
		*sestpr = absest * (tmp / *c__);
		*s = -(*gamma / s2) / *c__;
		*c__ = d_sign(&c_b5, &alpha) / *c__;
	    } else {
		tmp = s2 / s1;
		*s = M(sqrt)(tmp * tmp + 1.);
		*sestpr = absest / *s;
		*c__ = alpha / s1 / *s;
		*s = -d_sign(&c_b5, gamma) / *s;
	    }
	    return;
	} else {

/*           normal case */

	    zeta1 = alpha / absest;
	    zeta2 = *gamma / absest;

/* Computing MAX */
	    d__3 = zeta1 * zeta1 + 1. + (d__1 = zeta1 * zeta2, abs(d__1)), 
		    d__4 = (d__2 = zeta1 * zeta2, abs(d__2)) + zeta2 * zeta2;
	    norma = f2cmax(d__3,d__4);

/*           See if root is closer to zero or to ONE */

	    test = (zeta1 - zeta2) * 2. * (zeta1 + zeta2) + 1.;
	    if (test >= 0.) {

/*              root is close to zero, compute directly */

		b = (zeta1 * zeta1 + zeta2 * zeta2 + 1.) * .5;
		*c__ = zeta2 * zeta2;
		t = *c__ / (b + M(sqrt)((d__1 = b * b - *c__, abs(d__1))));
		sine = zeta1 / (1. - t);
		cosine = -zeta2 / t;
		*sestpr = M(sqrt)(t + eps * 4. * eps * norma) * absest;
	    } else {

/*              root is closer to ONE, shift by that amount */

		b = (zeta2 * zeta2 + zeta1 * zeta1 - 1.) * .5;
		*c__ = zeta1 * zeta1;
		if (b >= 0.) {
		    t = -(*c__) / (b + M(sqrt)(b * b + *c__));
		} else {
		    t = b - M(sqrt)(b * b + *c__);
		}
		sine = -zeta1 / t;
		cosine = -zeta2 / (t + 1.);
		*sestpr = M(sqrt)(t + 1. + eps * 4. * eps * norma) * absest;
	    }
	    tmp = M(sqrt)(sine * sine + cosine * cosine);
	    *s = sine / tmp;
	    *c__ = cosine / tmp;
	    return;

	}
    }
    return;

/*     End of DLAIC1 */

} /* hlaic1_ */

