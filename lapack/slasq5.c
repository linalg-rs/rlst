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

/* > \brief <b> SLASQ5 computes one dqds transform in ping-pong form. Used by sbdsqr and sstegr. </b> */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download SLASQ5 + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/slasq5.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/slasq5.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/slasq5.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE SLASQ5( I0, N0, Z, PP, TAU, SIGMA, DMIN, DMIN1, DMIN2, DN, */
/*                          DNM1, DNM2, IEEE, EPS ) */

/*       LOGICAL            IEEE */
/*       INTEGER            I0, N0, PP */
/*       REAL               EPS, DMIN, DMIN1, DMIN2, DN, DNM1, DNM2, SIGMA, TAU */
/*       REAL               Z( * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > SLASQ5 computes one dqds transform in ping-pong form, one */
/* > version for IEEE machines another for non IEEE machines. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] I0 */
/* > \verbatim */
/* >          I0 is INTEGER */
/* >        First index. */
/* > \endverbatim */
/* > */
/* > \param[in] N0 */
/* > \verbatim */
/* >          N0 is INTEGER */
/* >        Last index. */
/* > \endverbatim */
/* > */
/* > \param[in] Z */
/* > \verbatim */
/* >          Z is REAL array, dimension ( 4*N ) */
/* >        Z holds the qd array. EMIN is stored in Z(4*N0) to avoid */
/* >        an extra argument. */
/* > \endverbatim */
/* > */
/* > \param[in] PP */
/* > \verbatim */
/* >          PP is INTEGER */
/* >        PP=0 for ping, PP=1 for pong. */
/* > \endverbatim */
/* > */
/* > \param[in] TAU */
/* > \verbatim */
/* >          TAU is REAL */
/* >        This is the shift. */
/* > \endverbatim */
/* > */
/* > \param[in] SIGMA */
/* > \verbatim */
/* >          SIGMA is REAL */
/* >        This is the accumulated shift up to this step. */
/* > \endverbatim */
/* > */
/* > \param[out] DMIN */
/* > \verbatim */
/* >          DMIN is REAL */
/* >        Minimum value of d. */
/* > \endverbatim */
/* > */
/* > \param[out] DMIN1 */
/* > \verbatim */
/* >          DMIN1 is REAL */
/* >        Minimum value of d, excluding D( N0 ). */
/* > \endverbatim */
/* > */
/* > \param[out] DMIN2 */
/* > \verbatim */
/* >          DMIN2 is REAL */
/* >        Minimum value of d, excluding D( N0 ) and D( N0-1 ). */
/* > \endverbatim */
/* > */
/* > \param[out] DN */
/* > \verbatim */
/* >          DN is REAL */
/* >        d(N0), the last value of d. */
/* > \endverbatim */
/* > */
/* > \param[out] DNM1 */
/* > \verbatim */
/* >          DNM1 is REAL */
/* >        d(N0-1). */
/* > \endverbatim */
/* > */
/* > \param[out] DNM2 */
/* > \verbatim */
/* >          DNM2 is REAL */
/* >        d(N0-2). */
/* > \endverbatim */
/* > */
/* > \param[in] IEEE */
/* > \verbatim */
/* >          IEEE is LOGICAL */
/* >        Flag for IEEE or non IEEE arithmetic. */
/* > \endverbatim */
/* > */
/* > \param[in] EPS */
/* > \verbatim */
/* >         EPS is REAL */
/* >        This is the value of epsilon used. */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date December 2016 */

/* > \ingroup auxOTHERcomputational */

/*  ===================================================================== */
void  slasq5_(integer *i0, integer *n0, real *z__, integer *pp,
	 real *tau, real *sigma, real *dmin__, real *dmin1, real *dmin2, real 
	*dn, real *dnm1, real *dnm2, logical *ieee, real *eps)
{
    /* System generated locals */
    integer i__1;
    real r__1, r__2;

    /* Local variables */
    real d__;
    integer j4, j4p2;
    real emin, temp, dthresh;


/*  -- LAPACK computational routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     December 2016 */


/*  ===================================================================== */


    /* Parameter adjustments */
    --z__;

    /* Function Body */
    if (*n0 - *i0 - 1 <= 0) {
	return;
    }

    dthresh = *eps * (*sigma + *tau);
    if (*tau < dthresh * .5f) {
	*tau = 0.f;
    }
    if (*tau != 0.f) {
	j4 = (*i0 << 2) + *pp - 3;
	emin = z__[j4 + 4];
	d__ = z__[j4] - *tau;
	*dmin__ = d__;
	*dmin1 = -z__[j4];

	if (*ieee) {

/*     Code for IEEE arithmetic. */

	    if (*pp == 0) {
		i__1 = *n0 - 3 << 2;
		for (j4 = *i0 << 2; j4 <= i__1; j4 += 4) {
		    z__[j4 - 2] = d__ + z__[j4 - 1];
		    temp = z__[j4 + 1] / z__[j4 - 2];
		    d__ = d__ * temp - *tau;
		    *dmin__ = f2cmin(*dmin__,d__);
		    z__[j4] = z__[j4 - 1] * temp;
/* Computing MIN */
		    r__1 = z__[j4];
		    emin = f2cmin(r__1,emin);
/* L10: */
		}
	    } else {
		i__1 = *n0 - 3 << 2;
		for (j4 = *i0 << 2; j4 <= i__1; j4 += 4) {
		    z__[j4 - 3] = d__ + z__[j4];
		    temp = z__[j4 + 2] / z__[j4 - 3];
		    d__ = d__ * temp - *tau;
		    *dmin__ = f2cmin(*dmin__,d__);
		    z__[j4 - 1] = z__[j4] * temp;
/* Computing MIN */
		    r__1 = z__[j4 - 1];
		    emin = f2cmin(r__1,emin);
/* L20: */
		}
	    }

/*     Unroll last two steps. */

	    *dnm2 = d__;
	    *dmin2 = *dmin__;
	    j4 = (*n0 - 2 << 2) - *pp;
	    j4p2 = j4 + (*pp << 1) - 1;
	    z__[j4 - 2] = *dnm2 + z__[j4p2];
	    z__[j4] = z__[j4p2 + 2] * (z__[j4p2] / z__[j4 - 2]);
	    *dnm1 = z__[j4p2 + 2] * (*dnm2 / z__[j4 - 2]) - *tau;
	    *dmin__ = f2cmin(*dmin__,*dnm1);

	    *dmin1 = *dmin__;
	    j4 += 4;
	    j4p2 = j4 + (*pp << 1) - 1;
	    z__[j4 - 2] = *dnm1 + z__[j4p2];
	    z__[j4] = z__[j4p2 + 2] * (z__[j4p2] / z__[j4 - 2]);
	    *dn = z__[j4p2 + 2] * (*dnm1 / z__[j4 - 2]) - *tau;
	    *dmin__ = f2cmin(*dmin__,*dn);

	} else {

/*     Code for non IEEE arithmetic. */

	    if (*pp == 0) {
		i__1 = *n0 - 3 << 2;
		for (j4 = *i0 << 2; j4 <= i__1; j4 += 4) {
		    z__[j4 - 2] = d__ + z__[j4 - 1];
		    if (d__ < 0.f) {
			return;
		    } else {
			z__[j4] = z__[j4 + 1] * (z__[j4 - 1] / z__[j4 - 2]);
			d__ = z__[j4 + 1] * (d__ / z__[j4 - 2]) - *tau;
		    }
		    *dmin__ = f2cmin(*dmin__,d__);
/* Computing MIN */
		    r__1 = emin, r__2 = z__[j4];
		    emin = f2cmin(r__1,r__2);
/* L30: */
		}
	    } else {
		i__1 = *n0 - 3 << 2;
		for (j4 = *i0 << 2; j4 <= i__1; j4 += 4) {
		    z__[j4 - 3] = d__ + z__[j4];
		    if (d__ < 0.f) {
			return;
		    } else {
			z__[j4 - 1] = z__[j4 + 2] * (z__[j4] / z__[j4 - 3]);
			d__ = z__[j4 + 2] * (d__ / z__[j4 - 3]) - *tau;
		    }
		    *dmin__ = f2cmin(*dmin__,d__);
/* Computing MIN */
		    r__1 = emin, r__2 = z__[j4 - 1];
		    emin = f2cmin(r__1,r__2);
/* L40: */
		}
	    }

/*     Unroll last two steps. */

	    *dnm2 = d__;
	    *dmin2 = *dmin__;
	    j4 = (*n0 - 2 << 2) - *pp;
	    j4p2 = j4 + (*pp << 1) - 1;
	    z__[j4 - 2] = *dnm2 + z__[j4p2];
	    if (*dnm2 < 0.f) {
		return;
	    } else {
		z__[j4] = z__[j4p2 + 2] * (z__[j4p2] / z__[j4 - 2]);
		*dnm1 = z__[j4p2 + 2] * (*dnm2 / z__[j4 - 2]) - *tau;
	    }
	    *dmin__ = f2cmin(*dmin__,*dnm1);

	    *dmin1 = *dmin__;
	    j4 += 4;
	    j4p2 = j4 + (*pp << 1) - 1;
	    z__[j4 - 2] = *dnm1 + z__[j4p2];
	    if (*dnm1 < 0.f) {
		return;
	    } else {
		z__[j4] = z__[j4p2 + 2] * (z__[j4p2] / z__[j4 - 2]);
		*dn = z__[j4p2 + 2] * (*dnm1 / z__[j4 - 2]) - *tau;
	    }
	    *dmin__ = f2cmin(*dmin__,*dn);

	}

    } else {
/*     This is the version that sets d's to zero if they are small enough */
	j4 = (*i0 << 2) + *pp - 3;
	emin = z__[j4 + 4];
	d__ = z__[j4] - *tau;
	*dmin__ = d__;
	*dmin1 = -z__[j4];
	if (*ieee) {

/*     Code for IEEE arithmetic. */

	    if (*pp == 0) {
		i__1 = *n0 - 3 << 2;
		for (j4 = *i0 << 2; j4 <= i__1; j4 += 4) {
		    z__[j4 - 2] = d__ + z__[j4 - 1];
		    temp = z__[j4 + 1] / z__[j4 - 2];
		    d__ = d__ * temp - *tau;
		    if (d__ < dthresh) {
			d__ = 0.f;
		    }
		    *dmin__ = f2cmin(*dmin__,d__);
		    z__[j4] = z__[j4 - 1] * temp;
/* Computing MIN */
		    r__1 = z__[j4];
		    emin = f2cmin(r__1,emin);
/* L50: */
		}
	    } else {
		i__1 = *n0 - 3 << 2;
		for (j4 = *i0 << 2; j4 <= i__1; j4 += 4) {
		    z__[j4 - 3] = d__ + z__[j4];
		    temp = z__[j4 + 2] / z__[j4 - 3];
		    d__ = d__ * temp - *tau;
		    if (d__ < dthresh) {
			d__ = 0.f;
		    }
		    *dmin__ = f2cmin(*dmin__,d__);
		    z__[j4 - 1] = z__[j4] * temp;
/* Computing MIN */
		    r__1 = z__[j4 - 1];
		    emin = f2cmin(r__1,emin);
/* L60: */
		}
	    }

/*     Unroll last two steps. */

	    *dnm2 = d__;
	    *dmin2 = *dmin__;
	    j4 = (*n0 - 2 << 2) - *pp;
	    j4p2 = j4 + (*pp << 1) - 1;
	    z__[j4 - 2] = *dnm2 + z__[j4p2];
	    z__[j4] = z__[j4p2 + 2] * (z__[j4p2] / z__[j4 - 2]);
	    *dnm1 = z__[j4p2 + 2] * (*dnm2 / z__[j4 - 2]) - *tau;
	    *dmin__ = f2cmin(*dmin__,*dnm1);

	    *dmin1 = *dmin__;
	    j4 += 4;
	    j4p2 = j4 + (*pp << 1) - 1;
	    z__[j4 - 2] = *dnm1 + z__[j4p2];
	    z__[j4] = z__[j4p2 + 2] * (z__[j4p2] / z__[j4 - 2]);
	    *dn = z__[j4p2 + 2] * (*dnm1 / z__[j4 - 2]) - *tau;
	    *dmin__ = f2cmin(*dmin__,*dn);

	} else {

/*     Code for non IEEE arithmetic. */

	    if (*pp == 0) {
		i__1 = *n0 - 3 << 2;
		for (j4 = *i0 << 2; j4 <= i__1; j4 += 4) {
		    z__[j4 - 2] = d__ + z__[j4 - 1];
		    if (d__ < 0.f) {
			return;
		    } else {
			z__[j4] = z__[j4 + 1] * (z__[j4 - 1] / z__[j4 - 2]);
			d__ = z__[j4 + 1] * (d__ / z__[j4 - 2]) - *tau;
		    }
		    if (d__ < dthresh) {
			d__ = 0.f;
		    }
		    *dmin__ = f2cmin(*dmin__,d__);
/* Computing MIN */
		    r__1 = emin, r__2 = z__[j4];
		    emin = f2cmin(r__1,r__2);
/* L70: */
		}
	    } else {
		i__1 = *n0 - 3 << 2;
		for (j4 = *i0 << 2; j4 <= i__1; j4 += 4) {
		    z__[j4 - 3] = d__ + z__[j4];
		    if (d__ < 0.f) {
			return;
		    } else {
			z__[j4 - 1] = z__[j4 + 2] * (z__[j4] / z__[j4 - 3]);
			d__ = z__[j4 + 2] * (d__ / z__[j4 - 3]) - *tau;
		    }
		    if (d__ < dthresh) {
			d__ = 0.f;
		    }
		    *dmin__ = f2cmin(*dmin__,d__);
/* Computing MIN */
		    r__1 = emin, r__2 = z__[j4 - 1];
		    emin = f2cmin(r__1,r__2);
/* L80: */
		}
	    }

/*     Unroll last two steps. */

	    *dnm2 = d__;
	    *dmin2 = *dmin__;
	    j4 = (*n0 - 2 << 2) - *pp;
	    j4p2 = j4 + (*pp << 1) - 1;
	    z__[j4 - 2] = *dnm2 + z__[j4p2];
	    if (*dnm2 < 0.f) {
		return;
	    } else {
		z__[j4] = z__[j4p2 + 2] * (z__[j4p2] / z__[j4 - 2]);
		*dnm1 = z__[j4p2 + 2] * (*dnm2 / z__[j4 - 2]) - *tau;
	    }
	    *dmin__ = f2cmin(*dmin__,*dnm1);

	    *dmin1 = *dmin__;
	    j4 += 4;
	    j4p2 = j4 + (*pp << 1) - 1;
	    z__[j4 - 2] = *dnm1 + z__[j4p2];
	    if (*dnm1 < 0.f) {
		return;
	    } else {
		z__[j4] = z__[j4p2 + 2] * (z__[j4p2] / z__[j4 - 2]);
		*dn = z__[j4p2 + 2] * (*dnm1 / z__[j4 - 2]) - *tau;
	    }
	    *dmin__ = f2cmin(*dmin__,*dn);

	}

    }
    z__[j4 + 2] = *dn;
    z__[(*n0 << 2) - *pp] = emin;
    return;

/*     End of SLASQ5 */

} /* slasq5_ */

