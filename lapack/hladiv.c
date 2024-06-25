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

/* > \brief \b DLADIV performs complex division in doublereal arithmetic, avoiding unnecessary overflow. */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download DLADIV + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dladiv.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dladiv.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dladiv.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE DLADIV( A, B, C, D, P, Q ) */

/*       DOUBLE PRECISION   A, B, C, D, P, Q */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > DLADIV performs complex division in  doublereal arithmetic */
/* > */
/* >                       a + i*b */
/* >            p + i*q = --------- */
/* >                       c + i*d */
/* > */
/* > The algorithm is due to Michael Baudin and Robert L. Smith */
/* > and can be found in the paper */
/* > "A Robust Complex Division in Scilab" */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] A */
/* > \verbatim */
/* >          A is DOUBLE PRECISION */
/* > \endverbatim */
/* > */
/* > \param[in] B */
/* > \verbatim */
/* >          B is DOUBLE PRECISION */
/* > \endverbatim */
/* > */
/* > \param[in] C */
/* > \verbatim */
/* >          C is DOUBLE PRECISION */
/* > \endverbatim */
/* > */
/* > \param[in] D */
/* > \verbatim */
/* >          D is DOUBLE PRECISION */
/* >          The scalars a, b, c, and d in the above expression. */
/* > \endverbatim */
/* > */
/* > \param[out] P */
/* > \verbatim */
/* >          P is DOUBLE PRECISION */
/* > \endverbatim */
/* > */
/* > \param[out] Q */
/* > \verbatim */
/* >          Q is DOUBLE PRECISION */
/* >          The scalars p and q in the above expression. */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date January 2013 */

/* > \ingroup doubleOTHERauxiliary */

/*  ===================================================================== */
void  hladiv_(halfreal *a, halfreal *b, halfreal *c__, 
	halfreal *d__, halfreal *p, halfreal *q)
{
    /* System generated locals */
    halfreal d__1, d__2;

    /* Local variables */
    halfreal s, aa, ab, bb, cc, cd, dd, be, un, ov, eps;
    extern halfreal hlamch_(char *);
    extern void  hladiv1_(halfreal *, halfreal *, 
	    halfreal *, halfreal *, halfreal *, halfreal *);


/*  -- LAPACK auxiliary routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     January 2013 */


/*  ===================================================================== */



    aa = *a;
    bb = *b;
    cc = *c__;
    dd = *d__;
/* Computing MAX */
    d__1 = abs(*a), d__2 = abs(*b);
    ab = f2cmax(d__1,d__2);
/* Computing MAX */
    d__1 = abs(*c__), d__2 = abs(*d__);
    cd = f2cmax(d__1,d__2);
    s = 1.;
    ov = hlamch_("Overflow threshold");
    un = hlamch_("Safe minimum");
    eps = hlamch_("Epsilon");
    be = 2. / (eps * eps);
    if (ab >= ov * .5) {
	aa *= .5;
	bb *= .5;
	s *= 2.;
    }
    if (cd >= ov * .5) {
	cc *= .5;
	dd *= .5;
	s *= .5;
    }
    if (ab <= un * 2. / eps) {
	aa *= be;
	bb *= be;
	s /= be;
    }
    if (cd <= un * 2. / eps) {
	cc *= be;
	dd *= be;
	s *= be;
    }
    if (abs(*d__) <= abs(*c__)) {
	hladiv1_(&aa, &bb, &cc, &dd, p, q);
    } else {
	hladiv1_(&bb, &aa, &dd, &cc, p, q);
	*q = -(*q);
    }
    *p *= s;
    *q *= s;

    return;

/*     End of DLADIV */

} /* hladiv_ */

/* > \ingroup doubleOTHERauxiliary */
void  hladiv1_(halfreal *a, halfreal *b, halfreal *c__, 
	halfreal *d__, halfreal *p, halfreal *q)
{
    halfreal r__, t;
    extern halfreal hladiv2_(halfreal *, halfreal *, halfreal *, 
	    halfreal *, halfreal *, halfreal *);


/*  -- LAPACK auxiliary routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     January 2013 */


/*  ===================================================================== */



    r__ = *d__ / *c__;
    t = 1. / (*c__ + *d__ * r__);
    *p = hladiv2_(a, b, c__, d__, &r__, &t);
    *a = -(*a);
    *q = hladiv2_(b, a, c__, d__, &r__, &t);

    return;

/*     End of DLADIV1 */

} /* hladiv1_ */

/* > \ingroup doubleOTHERauxiliary */
halfreal hladiv2_(halfreal *a, halfreal *b, halfreal *c__, halfreal 
	*d__, halfreal *r__, halfreal *t)
{
    /* System generated locals */
    halfreal ret_val;

    /* Local variables */
    halfreal br;


/*  -- LAPACK auxiliary routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     January 2013 */


/*  ===================================================================== */



    if (*r__ != 0.) {
	br = *b * *r__;
	if (br != 0.) {
	    ret_val = (*a + br) * *t;
	} else {
	    ret_val = *a * *t + *b * *t * *r__;
	}
    } else {
	ret_val = (*a + *d__ * (*b / *c__)) * *t;
    }

    return ret_val;

/*     End of DLADIV12 */

} /* hladiv2_ */

