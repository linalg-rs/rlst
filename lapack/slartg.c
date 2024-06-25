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

/* > \brief \b SLARTG generates a plane rotation with real cosine and real sine. */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download SLARTG + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/slartg.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/slartg.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/slartg.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE SLARTG( F, G, CS, SN, R ) */

/*       REAL               CS, F, G, R, SN */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > SLARTG generate a plane rotation so that */
/* > */
/* >    [  CS  SN  ]  .  [ F ]  =  [ R ]   where CS**2 + SN**2 = 1. */
/* >    [ -SN  CS  ]     [ G ]     [ 0 ] */
/* > */
/* > This is a slower, more accurate version of the BLAS1 routine SROTG, */
/* > with the following other differences: */
/* >    F and G are unchanged on return. */
/* >    If G=0, then CS=1 and SN=0. */
/* >    If F=0 and (G .ne. 0), then CS=0 and SN=1 without doing any */
/* >       floating point operations (saves work in SBDSQR when */
/* >       there are zeros on the diagonal). */
/* > */
/* > If F exceeds G in magnitude, CS will be positive. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] F */
/* > \verbatim */
/* >          F is REAL */
/* >          The first component of vector to be rotated. */
/* > \endverbatim */
/* > */
/* > \param[in] G */
/* > \verbatim */
/* >          G is REAL */
/* >          The second component of vector to be rotated. */
/* > \endverbatim */
/* > */
/* > \param[out] CS */
/* > \verbatim */
/* >          CS is REAL */
/* >          The cosine of the rotation. */
/* > \endverbatim */
/* > */
/* > \param[out] SN */
/* > \verbatim */
/* >          SN is REAL */
/* >          The sine of the rotation. */
/* > \endverbatim */
/* > */
/* > \param[out] R */
/* > \verbatim */
/* >          R is REAL */
/* >          The nonzero component of the rotated vector. */
/* > */
/* >  This version has a few statements commented out for thread safety */
/* >  (machine parameters are computed on each entry). 10 feb 03, SJH. */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date December 2016 */

/* > \ingroup OTHERauxiliary */

/*  ===================================================================== */
void  slartg_(real *f, real *g, real *cs, real *sn, real *r__)
{
    /* System generated locals */
    integer i__1;
    real r__1, r__2;

    /* Local variables */
    integer i__;
    real f1, g1, eps, scale;
    integer count;
    real safmn2, safmx2;
    extern real slamch_(char *);
    real safmin;


/*  -- LAPACK auxiliary routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     December 2016 */


/*  ===================================================================== */

/*     LOGICAL            FIRST */
/*     SAVE               FIRST, SAFMX2, SAFMIN, SAFMN2 */
/*     DATA               FIRST / .TRUE. / */

/*     IF( FIRST ) THEN */
    safmin = slamch_("S");
    eps = slamch_("E");
    r__1 = slamch_("B");
    i__1 = (integer) (M(log)(safmin / eps) / M(log)(slamch_("B")) / 2.f);
    safmn2 = pow_ri(&r__1, &i__1);
    safmx2 = 1.f / safmn2;
/*        FIRST = .FALSE. */
/*     END IF */
    if (*g == 0.f) {
	*cs = 1.f;
	*sn = 0.f;
	*r__ = *f;
    } else if (*f == 0.f) {
	*cs = 0.f;
	*sn = 1.f;
	*r__ = *g;
    } else {
	f1 = *f;
	g1 = *g;
/* Computing MAX */
	r__1 = abs(f1), r__2 = abs(g1);
	scale = f2cmax(r__1,r__2);
	if (scale >= safmx2) {
	    count = 0;
L10:
	    ++count;
	    f1 *= safmn2;
	    g1 *= safmn2;
/* Computing MAX */
	    r__1 = abs(f1), r__2 = abs(g1);
	    scale = f2cmax(r__1,r__2);
	    if (scale >= safmx2) {
		goto L10;
	    }
/* Computing 2nd power */
	    r__1 = f1;
/* Computing 2nd power */
	    r__2 = g1;
	    *r__ = M(sqrt)(r__1 * r__1 + r__2 * r__2);
	    *cs = f1 / *r__;
	    *sn = g1 / *r__;
	    i__1 = count;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		*r__ *= safmx2;
/* L20: */
	    }
	} else if (scale <= safmn2) {
	    count = 0;
L30:
	    ++count;
	    f1 *= safmx2;
	    g1 *= safmx2;
/* Computing MAX */
	    r__1 = abs(f1), r__2 = abs(g1);
	    scale = f2cmax(r__1,r__2);
	    if (scale <= safmn2) {
		goto L30;
	    }
/* Computing 2nd power */
	    r__1 = f1;
/* Computing 2nd power */
	    r__2 = g1;
	    *r__ = M(sqrt)(r__1 * r__1 + r__2 * r__2);
	    *cs = f1 / *r__;
	    *sn = g1 / *r__;
	    i__1 = count;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		*r__ *= safmn2;
/* L40: */
	    }
	} else {
/* Computing 2nd power */
	    r__1 = f1;
/* Computing 2nd power */
	    r__2 = g1;
	    *r__ = M(sqrt)(r__1 * r__1 + r__2 * r__2);
	    *cs = f1 / *r__;
	    *sn = g1 / *r__;
	}
	if (abs(*f) > abs(*g) && *cs < 0.f) {
	    *cs = -(*cs);
	    *sn = -(*sn);
	    *r__ = -(*r__);
	}
    }
    return;

/*     End of SLARTG */

} /* slartg_ */

