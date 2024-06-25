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

/* > \brief \b DLARTGS generates a plane rotation designed to introduce a bulge in implicit QR iteration for t
he bidiagonal SVD problem. */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download DLARTGS + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlartgs
.f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlartgs
.f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlartgs
.f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE DLARTGS( X, Y, SIGMA, CS, SN ) */

/*       DOUBLE PRECISION        CS, SIGMA, SN, X, Y */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > DLARTGS generates a plane rotation designed to introduce a bulge in */
/* > Golub-Reinsch-style implicit QR iteration for the bidiagonal SVD */
/* > problem. X and Y are the top-row entries, and SIGMA is the shift. */
/* > The computed CS and SN define a plane rotation satisfying */
/* > */
/* >    [  CS  SN  ]  .  [ X^2 - SIGMA ]  =  [ R ], */
/* >    [ -SN  CS  ]     [    X * Y    ]     [ 0 ] */
/* > */
/* > with R nonnegative.  If X^2 - SIGMA and X * Y are 0, then the */
/* > rotation is by PI/2. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] X */
/* > \verbatim */
/* >          X is DOUBLE PRECISION */
/* >          The (1,1) entry of an upper bidiagonal matrix. */
/* > \endverbatim */
/* > */
/* > \param[in] Y */
/* > \verbatim */
/* >          Y is DOUBLE PRECISION */
/* >          The (1,2) entry of an upper bidiagonal matrix. */
/* > \endverbatim */
/* > */
/* > \param[in] SIGMA */
/* > \verbatim */
/* >          SIGMA is DOUBLE PRECISION */
/* >          The shift. */
/* > \endverbatim */
/* > */
/* > \param[out] CS */
/* > \verbatim */
/* >          CS is DOUBLE PRECISION */
/* >          The cosine of the rotation. */
/* > \endverbatim */
/* > */
/* > \param[out] SN */
/* > \verbatim */
/* >          SN is DOUBLE PRECISION */
/* >          The sine of the rotation. */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date November 2017 */

/* > \ingroup auxOTHERcomputational */

/*  ===================================================================== */
void  hlartgs_(halfreal *x, halfreal *y, halfreal *sigma,
	 halfreal *cs, halfreal *sn)
{
    halfreal r__, s, w, z__;
    extern halfreal hlamch_(char *);
    halfreal thresh;
    extern void  hlartgp_(halfreal *, halfreal *, 
	    halfreal *, halfreal *, halfreal *);


/*  -- LAPACK computational routine (version 3.8.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     November 2017 */


/*  =================================================================== */


    thresh = hlamch_("E");

/*     Compute the first column of B**T*B - SIGMA^2*I, up to a scale */
/*     factor. */

    if (*sigma == 0. && abs(*x) < thresh || abs(*x) == *sigma && *y == 0.) {
	z__ = 0.;
	w = 0.;
    } else if (*sigma == 0.) {
	if (*x >= 0.) {
	    z__ = *x;
	    w = *y;
	} else {
	    z__ = -(*x);
	    w = -(*y);
	}
    } else if (abs(*x) < thresh) {
	z__ = -(*sigma) * *sigma;
	w = 0.;
    } else {
	if (*x >= 0.) {
	    s = 1.;
	} else {
	    s = -1.;
	}
	z__ = s * (abs(*x) - *sigma) * (s + *sigma / *x);
	w = s * *y;
    }

/*     Generate the rotation. */
/*     CALL DLARTGP( Z, W, CS, SN, R ) might seem more natural; */
/*     reordering the arguments ensures that if Z = 0 then the rotation */
/*     is by PI/2. */

    hlartgp_(&w, &z__, sn, cs, &r__);

    return;

/*     End DLARTGS */

} /* hlartgs_ */

